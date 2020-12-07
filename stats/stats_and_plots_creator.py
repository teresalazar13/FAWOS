import time
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import itertools
from typing import List

from models.Label import Label
from models.dataset import Dataset
from taxonomizing.Taxonomy import Taxonomy
from stats.Distribution import Distribution
from taxonomizing.TaxonomyAndNeighbours import TaxonomyAndNeighbours


def save_stats(dataset: Dataset,
               df: pd.DataFrame,
               taxonomies_and_neighbours: List[TaxonomyAndNeighbours],
               stats_filename: str) -> list:

    labels, str_labels = get_labels(dataset, df, taxonomies_and_neighbours)
    Distribution.save_distributions(labels, stats_filename)

    return str_labels


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
def create_points_plot(labels: List,
                       df: pd.DataFrame,
                       plot_filename_train: str,
                       plot_filename_oversampled: str,
                       number_of_train_points: int):
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=2, perplexity=40, n_iter=250, learning_rate=1000)
    data = df[df.columns].values
    tsne_results = tsne.fit_transform(data)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    df_tsne = df.copy(deep=True)
    df_tsne['tsne-2d-one'] = tsne_results[:, 0]
    df_tsne['tsne-2d-two'] = tsne_results[:, 1]

    save_points_plot(df_tsne.head(number_of_train_points), labels[: number_of_train_points], plot_filename_train)
    save_points_plot(df_tsne, labels, plot_filename_oversampled)


def save_points_plot(df_tsne, labels, plot_filename):
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


def create_distributions_plot(dataset: Dataset):
    train_distributions = dataset.get_train_distributions()
    oversampled_distributions = dataset.get_oversampled_distributions()
    possible_combinations = create_possible_class_combinations(dataset)
    labels = []
    train_percentages = []
    oversampled_percentages = []

    for comb in possible_combinations:
        train_value = 0
        oversampled_value = 0
        d = None
        for train_distribution in train_distributions:
            if comb_matches_distribution(dataset, comb, train_distribution):
                train_value = train_distribution.percentage
                d = train_distribution
                break
        for oversampled_distribution in oversampled_distributions:
            if comb_matches_distribution(dataset, comb, oversampled_distribution):
                oversampled_value = oversampled_distribution.percentage
                d = oversampled_distribution
                break

        if d:
            labels.append(d.label.target_class_value[0] + " "
                          + " ".join(d.label.sensitive_class_values.values()) + " "
                          + d.label.taxonomy.value[0])
            train_percentages.append(train_value)
            oversampled_percentages.append(oversampled_value)

    x = np.arange(len(labels))  # the label locations
    width = 0.40  # the width of the bars

    fig, ax = plt.subplots(figsize=(len(labels) * 1.3, 5))
    rects1 = ax.bar(x - width / 2, train_percentages, width, label='Train')
    rects2 = ax.bar(x + width / 2, oversampled_percentages, width, label='Oversampled')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('%')
    ax.set_title('% by classes')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    plt.savefig(dataset.get_distributions_plot_filename())


def comb_matches_distribution(dataset, comb, distribution):
    if comb[dataset.target_class.name] == distribution.label.target_class_value:
        for class_name, value in distribution.label.sensitive_class_values.items():
            if comb[class_name] != [value]:  # TODO replace when multiclass
                return False

        if comb["taxonomy"] != distribution.label.taxonomy:
            return False

        return True

    return False


def create_possible_class_combinations(dataset: Dataset):
    classes = [
        [{dataset.target_class.name: dataset.target_class.positive_class},
        {dataset.target_class.name: dataset.target_class.negative_class}]
    ]
    for sensitive_class in dataset.sensitive_classes:
        classes.append(
            [
                {sensitive_class.name: sensitive_class.privileged_classes},
                {sensitive_class.name: sensitive_class.unprivileged_classes}
            ]
        )
    classes.append(
        [
            {"taxonomy": Taxonomy.OUTLIER},
            {"taxonomy": Taxonomy.RARE},
            {"taxonomy": Taxonomy.BORDERLINE},
            {"taxonomy": Taxonomy.SAFE}
        ]
    )

    combs_tuples = itertools.product(*classes)
    combs = []
    for comb_tuples in combs_tuples:
        comb = {}
        for d in comb_tuples:
            comb.update(d)
        combs.append(comb)

    return combs
