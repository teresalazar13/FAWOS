import time
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from typing import List

from models.Label import Label
from models.dataset import Dataset
from stats.Distribution import Distribution
from taxonomizing.TaxonomyAndNeighbours import TaxonomyAndNeighbours


def save_distributions_and_plots(dataset: Dataset,
                                 df: pd.DataFrame,
                                 taxonomies_and_neighbours: List[TaxonomyAndNeighbours],
                                 stats_filename: str,
                                 plot_filename: str):

    labels, str_labels = get_labels(dataset, df, taxonomies_and_neighbours)
    save_plot_tsne(str_labels, df, plot_filename)
    Distribution.save_distributions(labels, stats_filename)


def get_labels(dataset: Dataset, df: pd.DataFrame, taxonomies_and_neighbours: List[TaxonomyAndNeighbours]) -> (List[Label], list):
    labels = []
    str_labels = []

    for i in range(len(taxonomies_and_neighbours)):
        dataset_mappings = dataset.get_dataset_mappings_inverted()
        target_class_value = df[dataset.target_class.name].iloc[i]
        target_class = dataset_mappings[dataset.target_class.name][target_class_value]

        sensitive_class_values = {}
        str_label = target_class

        for sensitive_class in dataset.sensitive_classes:
            value = df[sensitive_class.name].iloc[i]
            original_value = dataset_mappings[sensitive_class.name][value]
            sensitive_class_values[sensitive_class.name] = original_value
            str_label += " " + original_value

        taxonomy = taxonomies_and_neighbours[i].taxonomy
        label = Label(target_class, sensitive_class_values, taxonomy)
        labels.append(label)
        str_labels.append(str_label)

    return labels, str_labels


# https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
def save_plot_tsne(labels: List, df: pd.DataFrame, plot_filename: str):
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=2, perplexity=40, n_iter=250, learning_rate=750)
    data = df[df.columns].values
    tsne_results = tsne.fit_transform(data)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    df_tsne = df.copy(deep=True)
    df_tsne['tsne-2d-one'] = tsne_results[:, 0]
    df_tsne['tsne-2d-two'] = tsne_results[:, 1]

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one",
        y="tsne-2d-two",
        palette=sns.color_palette("hls", len(set(labels))),
        data=df_tsne,
        hue=labels,
        legend="full",
        alpha=0.3
    )

    plt.savefig(plot_filename)
