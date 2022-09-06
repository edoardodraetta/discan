from shiny import App, render, ui, reactive
from pyrsistent import pset, pvector # immutable objects
# import matplotlib.pyplot as plt
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
model_probas = pd.read_csv("data/compr_probas_2022_0.csv.gz", index_col=0, compression='gzip')
for chunk in range(1, 10):
    filename = "data/compr_probas_2022_%02d.csv.gz" % chunk
    model_probas = pd.concat(
        [model_probas, pd.read_csv(filename, index_col=0, compression='gzip')],
        axis=1
)

# model_probas = pd.read_csv("./data/pheno_new_probas.csv", index_col=0)

model_ranks = model_probas.rank(axis=1, method="min", pct=True, ascending=False)
model_ranks.index = [str(x) for x in model_ranks.index]


app_ui = ui.page_fluid(
    ui.panel_title("Welcome to the DisCan testing environment"),
    ui.layout_sidebar(
        ui.panel_sidebar(
# Options
            ui.input_switch("option", "Option?", value=False),
            ui.input_radio_buttons(
                "year", "Choose an HPO year:",
                {"2019": "2019", "2022": "2022",},
            ),
# Genes
                ui.input_text(
                    "genes_list",
                    "Enter list of genes (separated by spaces):",
                    "TP53 BRCA1 BRCA2 VEGFA TP53 bLa test"),
                ui.output_text_verbatim("genes_report"),
# Diseases
                ui.input_selectize(
                    "disease_selected",
                    "Choose a disease:",
                    choices=diseases,
                    selected="",
                ),

                ui.input_action_button("add", "Add phenotypes associated to disease"),
# HPO Terms
                ui.input_selectize(
                    "hpo_selected",
                    "Choose some phenotypes:",
                    choices=hpo_terms,
                    selected='',
                    multiple=True,
                ),
# Go!
                ui.input_action_button("compute", "Compute!"),
                ui.input_action_button("reset", "Reset"),
        ),
# Output!
        ui.panel_main(
            ui.output_table("result"),
            ),
    )
)


def server(input, output, session):


# Handling Genes

    @reactive.Calc
    def get_genes():
        # Given the input genes list, returns valid gene Entrez IDs and Names
        # - TODO: cast string to uppercase

        genes = input.genes_list().split() # input genes
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

        return list(final_genes), list(genes_in)

    @output
    @render.text
    def genes_report():
        # Output valid gene names
        _, genes = get_genes()
        return f"Valid genes are\n {list(set(genes))}"

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

# Rank calculations and display
    @reactive.Calc
    def compute_gene_ranks():
        # Get input genes and hpo_terms
        genes, _ = get_genes()
        terms = get_hpo_terms()

        # No input (can add error reporting later, including for invalid genes)
        if len(genes) == 0 or len(terms) == 0:
            return None

        # Reduce HPO terms to include only HP ID
        terms = [x[:10] for x in terms]

        # Select only selected genes
        subset_ranks = model_ranks[genes] # subset data to perform less calculation

        # Selected terms that are in the dataset, or nearest ancestors
        subset_terms = reduce_terms(subset_ranks, hpo_net, terms)

        ranked_genes = combine_ranks(subset_ranks, subset_terms).sort_values()

        return ranked_genes, "subset_info"



    @output
    @render.table
    def result():
        if input.compute() != 0:
            with reactive.isolate():
                if compute_gene_ranks() == None:
                    return ""
                ranked_genes, subset_info = compute_gene_ranks()
                return pd.DataFrame(zip(ranked_genes.index, ranked_genes), columns=["Entrez ID", "Score"])

app = App(app_ui, server)