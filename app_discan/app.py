from shiny import App, render, ui, reactive
from pyrsistent import pset, pvector # immutable objects
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx

# Logscale version of the geometric mean
def gmean(data, axis=0):
    return np.exp(np.mean(np.log(data), axis=axis))


# get geometric mean of ranks for given terms
def combine_ranks(data, hpo_terms):

    # Make sure hpo_terms are in the data
    hpo_terms = [x for x in hpo_terms if x in data.index]

    # Take the vertical mean (across scores in each gene)
    # Label as genes

    return pd.Series(gmean(data.loc[hpo_terms, :], axis=0), index=data.columns)


# given a list of hpo terms, get the closest ancestor in dataset columns
def reduce_terms(data, hpo_tree, hpo_terms):
    term_list = set() # sets cannot contain duplicate values (sets are mutable)
    for term in hpo_terms:
        #get the list of terms that actually are in dataset
        #HP:0000118 is "phenotypic abnormality"
        #I'm not familiar with how networkx works, so I will trust this
        path = [x for x in nx.shortest_path(hpo_tree, term, 'HP:0000118') if x in data.columns]

        if len(path) > 0: # if we've moved towards HP:0000118, add to term list
            print("Phenotype not in model, finding closest ancestor.")
            term_list.add(path[0])
        else:
            term_list.add(term)
    # Return list, not set
    return list(term_list)


# Loading data

DATADIR = "app_discan/" # Empty for shinyapps.io

# Gene and disease annotations
gene_annotation = nx.read_graphml(DATADIR + "gene_annotation.graphml")
disease_net = nx.read_graphml(DATADIR + "disease_annot.graphml")

# HPO Terms
hpo_net = nx.read_graphml(DATADIR + "hp.220616.obo.graphml")
hpo_ids = [x for x in hpo_net.nodes() if nx.has_path(hpo_net, x, 'HP:0000118')]
hpo_ids.sort()
hpo_ids = hpo_ids[1:]
hpo_names = [hpo_net.nodes()[x]['name'] for x in hpo_ids]
hpo_terms = ["%s: %s" % (hpo_ids[x], hpo_names[x]) for x in range(len(hpo_ids))]

# Diseases
disease_ids = [x for x in disease_net.nodes() if disease_net.nodes()[x]['type'] == 'disease' and 'name' in disease_net.nodes()[x]]
disease_ids.sort()
disease_names = [disease_net.nodes()[x]['name'] for x in disease_ids if 'name' in disease_net.nodes()[x]]
diseases = ["%s: %s" % (disease_ids[x], disease_names[x]) for x in range(len(disease_ids))]

# Model

# Read from chunks
model_probas = pd.read_csv(DATADIR + "compr_probas_2022_00.csv.gz", index_col=0, compression='gzip')
for chunk in range(1, 10):
    filename = DATADIR + "compr_probas_2022_%02d.csv.gz" % chunk
    model_probas = pd.concat(
        [model_probas, pd.read_csv(filename, index_col=0, compression='gzip')],
        axis=1
)

# Create rank matrix
model_ranks = model_probas.rank(axis=1, method="min", pct=True, ascending=False)
model_ranks.index = [str(x) for x in model_ranks.index]

# UI
app_ui = ui.page_fluid(
    ui.panel_title("DISCAN"),
    ui.layout_sidebar(

        ui.panel_sidebar(
            ui.markdown("##### Welcome"),
            ui.markdown(
                "This application calculates the likelihood of involvement of one or " \
                "more genes in a genetic syndrome characterized by a set of abnormal phenotypes."
                ),

            ui.markdown("##### Enter a list of genes: "),
            ui.input_text_area(
                "genes_list",
                "",
                # "TP53\nBRCA1\nBRCA2\nVEGFA\nKDM6A\nTP53\nTNF\nbla\n.+0\nEGFR\nVEGFA\nAPOE\nIL6\nTGFBI\nMTHFR\nESR1\nAKT1",
                "",
                height='200px',
                ),

            ui.markdown("##### Select abnormal phenotypes:"),
            ui.input_selectize(
                "disease_selected",
                "Select a disease, then press add:",
                choices=diseases,
                selected=["OMIM:147920: KABUKI SYNDROME 1: KABUK1"], # Doesn't work for some reason
                width="400px",
                multiple=True
                ),
            ui.input_action_button(
                "add",
                "Add"
                ),
            ui.input_action_button(
                "clear",
                "Clear"
                ),
            ui.HTML("<br><br>"),
            ui.input_selectize(
                "hpo_selected",
                "OR, Manually select abnormal phenotypes:",
                choices=hpo_terms,
                selected='',
                multiple=True,
                width=200
                ),
            ),

        ui.panel_main(
            # ui.output_table("genes_report_table"),
            ui.input_switch("quantile_switch","Plot quantiles"),
            ui.input_action_button("compute", "Compute"),
            ui.markdown("#### Results"),
            ui.output_plot("plot"),
            ui.output_table("display_result"),
            ui.output_text_verbatim("warning"),
            # ui.HTML("<hr><b>Warning!</b> The following items were excluded from the analysis:<br>"),
        ),
    ),
)


def server(input, output, session):

# Genes
    @reactive.Calc
    def get_input_genes():

        # 1. Parse input
        genes = input.genes_list().split() # input genes
        genes = [x.upper() for x in genes] # cast to upper
        genes = list(set(genes)) # unique elements only

        # 2. Check if genes are in annotation
        genes_out = [g for g in genes if not g in gene_annotation.nodes()]
        if len(genes_out) != 0:
            # remove from genes to consider
            genes = [g for g in genes if not g in genes_out]

        df = pd.DataFrame(genes, columns=["Input"])

        # 3. Get Entrez IDs of considered genes

        # A long lambda function:
        # - if the input is already an Entrez ID, keep that.
        # - else, decode the graph at the given node to find the entrez ID

        df["Entrez ID"] = df["Input"].apply(lambda g: g if gene_annotation.nodes()[g]['type'] == 'entrez' else list(gene_annotation.neighbors(g))[0])

        # 4. Get gene symbols
        df["Gene Symbol"] = df["Entrez ID"].apply(lambda x: list(gene_annotation.neighbors(x))[0])

        # 5. Check that gene is in model
        df["In Model"] = df["Entrez ID"].apply(lambda x: "Y" if x in model_ranks.columns else "N")

        # 7. Reorder Columns
        df = df[['Input', 'Entrez ID', "Gene Symbol", 'In Model']]
        return df, genes_out

    @output
    @render.table
    # May choose to display the input genes dataframe on the page
    def genes_report_table():
        df, _= get_input_genes()
        return df

    @output
    @render.text
    @reactive.event(input.compute)
    def warning():

        df, invalid_input = get_input_genes()
        not_in_model = df[df["In Model"] == "N"]

        # Report excluded genes
        if len(invalid_input) == 0 and len(not_in_model) == 0:
            return ""

        spool = "Warning! The following genes were excluded from analysis:\n\n"
        if len(invalid_input) != 0:
            for gene in invalid_input:
                spool += f" - {gene} is not an annotated gene. \n"

        if len(not_in_model) != 0:
            for gene in not_in_model.iterrows():
                spool += f" - {gene[1][2]} (Entrez ID {gene[1][1]}) was not included in the model. \n"

        ui.notification_show("Warning! Some genes were excluded from analysis!", type="Warning")
        return spool

# Display HPO selection
    @reactive.Calc
    def decode_OMIM():

        # Get OMIM ID
        diseases = [d[:11] for d in input.disease_selected()]

        hp_sel = pvector()
        for dis in diseases:
            # Get hp terms
            hp_map = [x for x in disease_net.neighbors(dis) if nx.has_path(hpo_net, x, 'HP:0000118')]
            hp_names = [hpo_net.nodes()[x]['name'] for x in hp_map]

            # Append ID: PHENOTYPE to selection
            hp_map_name = pvector("%s: %s" % (hp_map[x], hp_names[x]) for x in range(len(hp_map)))
            hp_sel += hp_map_name

        return hp_sel

    @reactive.Calc
    @reactive.event(input.add)
    def get_hpo_terms():
        return set(decode_OMIM() + input.hpo_selected())

    @reactive.Effect()
    @reactive.event(input.add)
    # On Add
    def _():
        ui.update_selectize(
            "hpo_selected",
            label="OR: Manually select abnormal phenotypes:",
            choices=hpo_terms,
            selected=list(get_hpo_terms()),
        )

    @reactive.Effect()
    # On clear
    @reactive.event(input.clear)
    def _():
        ui.update_selectize(
            "hpo_selected",
            label="OR: Manually select abnormal phenotypes:",
            choices=hpo_terms,
            selected="",
        )


# Rank calculations
    @reactive.Calc
    @reactive.event(input.compute)
    def rank_all_genes():

        # Get input hpo_terms, slice to only HP ID
        terms = get_hpo_terms()

        if len(terms) == 0:
            ui.notification_show("Error: please enter at least one phenotype", type="error")
            return pd.Series(None)

        terms = [x[:10] for x in terms]

        # Selected terms that are in the dataset, or nearest ancestors
        subset_terms = reduce_terms(model_ranks, hpo_net, terms)

        # Perform geometric mean operation on all genes in model
        return combine_ranks(model_ranks, subset_terms)

    @reactive.Calc
    @reactive.event(input.compute)
    def calc_result():

        # Select input genes
        df, _ = get_input_genes()
        genes = list(df[df["In Model"]=="Y"]["Entrez ID"])

        # Perform geometric mean operation on all genes in model
        all_ranks = rank_all_genes()

        if len(all_ranks) == 0:
            return None

        # Calculate quantiles
        quantiles = pd.qcut(all_ranks, 5, labels=False)

        # Selected gene ranks, sorted by score
        sel_gene_ranks = all_ranks[genes]
        sel_gene_ranks.sort_values(ascending=True, inplace=True)
        sel_quantiles = quantiles[genes]
        sel_quantiles.sort_values(ascending=True, inplace=True)

        ids = []
        names = []
        symbols = []
        for gene in sel_gene_ranks.index:
            ids.append(gene)
            names.append(gene_annotation.nodes()[gene]['name'])
            symbols.append(list(gene_annotation.neighbors(gene))[0])

        columns = ["Entrez ID", "Gene Symbol", "Name and Description", "Score", "Quantile"]
        return pd.DataFrame(
            zip(ids, symbols, names, sel_gene_ranks, sel_quantiles+1),
            columns=columns)

    @output
    @render.table
    @reactive.event(input.compute)
    def display_result():
        return calc_result()

# Plot
    @reactive.event(input.compute)
    def make_plot():
        # Get results
        result = calc_result()
        all_ranks = rank_all_genes()

        if len(all_ranks) == 0:
            return None

        fig, ax = plt.subplots(figsize = (12, 7))

        # Histogram of all scores

        sns.histplot(rank_all_genes(), ax=ax, edgecolor='white')

        # Draw quantile lines
        if input.quantile_switch():
            q = ['25%','50%', '75%']
            colors = ['black', 'black', 'black']
            desc = all_ranks.describe()
            for i in range(len(q)):
                ax.axvline(desc[q[i]], color=colors[i], ls="--")

        # Draw stems with labels
        stem_y =  np.linspace(700, 1800, result.shape[0])
        plt.stem(
            result["Score"],
            stem_y,
            linefmt="C3:", markerfmt="C3.", basefmt="None"
            )

        for i in range(result.shape[0]):
            ax.annotate(
                result["Gene Symbol"][i],
                xy=(result["Score"][i], stem_y[i]+20),
                size=8
                )

        plt.ylim(0,2000)
        sns.despine()

        plt.title("Distribution of Gene Ranks of all Genes in Model", y=1.02, weight=1.2)

        return ax

    @output
    @render.plot
    @reactive.event(input.compute)
    def plot():
        ax = make_plot()
        return ax


app = App(app_ui, server)
