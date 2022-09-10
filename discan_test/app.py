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

# standard deviation from the geometric mean
def gstd(data, axis=0):
    mu = gmean(data, axis=axis)

    return np.exp(np.sqrt(np.mean((np.log(data) - mu)**2, axis=axis)))

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
gene_annotation = nx.read_graphml("data/gene_annotation.graphml")
disease_net = nx.read_graphml("data/disease_annot.graphml")
# HPO Terms
hpo_net = nx.read_graphml("data/hp.220616.obo.graphml") #the HPO obo tree (OLD!)
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
model_probas = pd.read_csv("data/compr_probas_2019_00.csv.gz", index_col=0, compression='gzip')
for chunk in range(1, 10):
    filename = "data/compr_probas_2022_%02d.csv.gz" % chunk
    model_probas = pd.concat(
        [model_probas, pd.read_csv(filename, index_col=0, compression='gzip')],
        axis=1
)

# model_probas = pd.read_csv("./data/pheno_new_probas.csv", index_col=0)

model_ranks = model_probas.rank(axis=1, method="min", pct=True, ascending=False)
model_ranks.index = [str(x) for x in model_ranks.index]

# UI
app_ui = ui.page_fluid(

    ui.panel_title("DISCAN"),

    ui.markdown("##### Welcome"),
    ui.markdown(
        "This application calculates the likelihood of involvement of one or " \
        "more genes in a genetic syndrome characterized by a set of abnormal phenotypes."
        ),

    ui.markdown(
        "(rewrite needed) The underlying model is a random forest classifier trained on gene features not related to gene function, "\
        "e.g. triplet frequency, coding sequence length, histone marks, and others. The two most predictive features, "\
        "however, were the evolutionary age of the gene and the number of times the gene was mutated across all cancers. " \
        "This second measure is the thrust of our paper, and reflects the fact that genes compatible with genetic disease " \
        "are likely to be compatible with cancer evolution ."
        ),

    ui.markdown("##### Abstract"),
    ui.markdown("Genomic sequence mutations can be pathogenic in both germline and somatic cells. Several authors have observed that often the same genes are involved in cancer when mutated in somatic cells and in genetic diseases when mutated in the germline. Recent advances in high-throughput sequencing techniques have provided us with large databases of both types of mutations, allowing us to investigate this issue in a systematic way. Hence, we applied a machine learning based framework to this problem, comparing multiple independent models. We found that genes characterized by high frequency of somatic mutations in the most common cancers and ancient evolutionary age are most likely to be involved in abnormal phenotypes and diseases. These results suggest that the combination of tolerance for mutations at the cell viability level (measured by the frequency of somatic mutations in cancer) and functional relevance (demonstrated by evolutionary conservation) are the main predictors of disease genes. Our results thus confirm the deep relationship between pathogenic mutations in somatic and germline cells, provide new insight into the common origin of cancer and genetic diseases, and can be used to improve the identification of new disease genes."),


    # ui.row(
    #     ui.column(6,
    #         ui.input_text_area(
    #             "genes_list",
    #             "Enter list of genes:",
    #             "TP53 \nBRCA1 \nBRCA2 \nVEGFA",
    #             height='200px',
    #             ),
    #         ),
    #     ui.column(6,
    #         ui.output_text_verbatim("genes_report"),
    #         )
    #     ),

    ui.markdown("## Ranking Genes with DISCAN"),

    ui.markdown("##### Enter a list of genes: "),
    ui.input_text_area(
        "genes_list",
        "",
        "TP53 \nBRCA1 \nBRCA2 \nVEGFA \nKDM6A",
        height='150px',
        ),
    ui.output_text_verbatim("genes_report"),

    ui.markdown("##### Enter a set of abnormal phenotypes:"),
    ui.markdown("Try selecting 'Kabuki Syndrome' using the default gene list."),
    ui.row(
        ui.column(6,
            ui.input_selectize(
                "hpo_selected",
                "Select abnormal phenotypes:",
                choices=hpo_terms,
                selected='',
                multiple=True,
                width=200
                ),
            ),
        ui.column(6,
            (
                ui.input_selectize(
                    "disease_selected",
                    "You may wish to select an OMIM disease which corresponds to a set of abnormal phenotypes.\n Select a disease, then press add.",
                    choices=diseases,
                    selected=[],
                    width="400px"
                    ),
                ui.input_action_button(
                    "add",
                    "Add"
                    ),
                )
            ),
        ),

    ui.input_action_button("compute", "Compute!"),
    ui.input_action_button("reset", "Reset"),

    ui.markdown("#### Results Table"),
    ui.output_table("result"),

    ui.markdown("#### Results Plot"),
    ui.output_plot("results_plot")
)



def server(input, output, session):

# Genes
    @reactive.Calc
    def get_genes():
        # Given the input genes list, returns valid gene Entrez IDs and Names

        genes = input.genes_list().split() # input genes
        genes = [x.upper() for x in genes] # cast to upper

        final_genes = set() # sets cannot contain duplicates

        # Select annotated and not annotated genes
        genes_in = [x for x in genes if x in gene_annotation.nodes()]
        genes_out = [x for x in genes if not x in gene_annotation.nodes()]

        # Find Entrez IDs or gene symbols
        for g in genes_in:
            if gene_annotation.nodes()[g]['type'] == 'entrez':
                final_genes.add(g)
            else:
                # a gene symbol is given
                [final_genes.add(x) for x in gene_annotation.neighbors(g)]

        return list(final_genes), list(set(genes_in)), list(set(genes_out))

    @output
    @render.text
    def genes_report():
        # Output valid gene names
        _, genes, invalid_genes = get_genes()

        if len(invalid_genes) == 0:
            return f"Valid genes are\n {genes}"
        else:
            return f"Valid genes are\n {genes}\nInvalid genes are\n {invalid_genes}"

# Display HPO selection
    @reactive.Calc
    def decode_OMIM():
        # Decode OMIM disease into terms
        # - TODO: Should happen only on add

        # Get OMIM ID
        dis_id = input.disease_selected()[:11]

        # Get hp terms
        hp_map = [x for x in disease_net.neighbors(dis_id) if nx.has_path(hpo_net, x, 'HP:0000118')]
        hp_names = [hpo_net.nodes()[x]['name'] for x in hp_map]

        # Return ID: PHENOTYPE
        # - Why does this need to be an immutable type?
        return pvector("%s: %s" % (hp_map[x], hp_names[x]) for x in range(len(hp_map)))


    @reactive.Calc # Make reactive value?
    def get_hpo_terms():
        # If add has been pressed, return the selected hpo terms plus
        # those included in the OMIM disease
        if input.add() != 0:
            return set(decode_OMIM() + input.hpo_selected())
        else:
            return input.hpo_selected()

    @reactive.Effect()
    def _():
        if input.add() != 0:
            with reactive.isolate(): # inside this block, don't depend on inputs!
                ui.update_selectize(
                    "hpo_selected",
                    label="Choose HPO terms",
                    choices=hpo_terms,
                    selected=list(get_hpo_terms()),
                )

# Rank calculations
    @output
    @render.table
    def result():
        input.compute()
        with reactive.isolate():

            if input.compute() == 0:
                return None

            # Get input genes and hpo_terms
            genes, _, _ = get_genes()
            terms = get_hpo_terms()

            # Reduce HPO terms to include only HP ID
            terms = [x[:10] for x in terms]

            # Subeset to selected genes
            # subset_ranks = model_ranks[genes] # subset data to perform less calculation

            # Selected terms that are in the dataset, or nearest ancestors
            subset_terms = reduce_terms(model_ranks, hpo_net, terms)

            # Perform geometric mean operation on all genes in model
            ranked_genes = combine_ranks(model_ranks, subset_terms)
            ranked_genes

            # Calculate quantiles
            quantiles = pd.qcut(ranked_genes, 5, labels=False)

            # Selected gene ranks
            sel_gene_ranks = ranked_genes[genes]
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
                zip(symbols, ids, names, sel_gene_ranks, sel_quantiles+1),
                columns=columns)



    @output
    @render.plot
    def results_plot():
        input.compute()
        with reactive.isolate():

            if input.compute() == 0:
                return None

            # 1. geometric mean of ranks of all genes in dataset over selected phenotypes

            # Get input genes and hpo_terms
            genes, _, _ = get_genes()
            terms = get_hpo_terms()

            # Reduce HPO terms to include only HP ID
            terms = [x[:10] for x in terms]

            # Selected terms that are in the dataset, or nearest ancestors
            subset_terms = reduce_terms(model_ranks, hpo_net, terms)

            # Perform geometric mean operation on all genes in model
            ranked_genes = combine_ranks(model_ranks, subset_terms)
            ranked_genes.sort_values(ascending=True, inplace=True)

            # Calculate quantiles
            # quantiles = pd.qcut(ranked_genes, 5, labels=False)
            # sel_quantiles = quantiles[genes]

            fig, ax = plt.subplots(figsize = (12, 5))

            sns.histplot(ranked_genes, ax=ax)

            q = ['25%','50%', '75%']
            colors = ['green', 'red', 'blue']
            desc = ranked_genes.describe()
            for i in range(len(q)):
                ax.axvline(desc[q[i]], color=colors[i], ls="--", label=q[i])

            ax.title.set_text("Histogram of Gene Ranks for all Genes in Model")
            ax.legend()

            return ax


app = App(app_ui, server)
