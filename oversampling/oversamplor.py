import operator
import random
import itertools
import numpy as np
import pandas as pd
from typing import List, Dict

from models.Feature import Feature
from models.FeatureTypeCategorical import FeatureTypeCategorical
from models.FeatureTypeContinuous import FeatureTypeContinuous
from models.FeatureTypeOrdinal import FeatureTypeOrdinal
from models.dataset import Dataset
from models.SensitiveClass import SensitiveClass
from oversampling.DatapointAndNeighbours import DatapointAndNeighbours
from oversampling.DatapointsFromClassToOversample import DatapointsFromClassToOversample
from oversampling.DatapointsToOversample import DatapointsToOversample
from taxonomizing.TaxonomyAndNeighbours import TaxonomyAndNeighbours
from taxonomizing.Taxonomy import Taxonomy
from stats.Distribution import Distribution


def oversample(dataset: Dataset,
               datapoints_from_class_to_oversample_list: List[DatapointsFromClassToOversample],
               safe_percentage: float,
               borderline_percentage: float,
               rare_percentage: float):

    random.seed(dataset.seed)
    df = dataset.get_train_dataset()

    for datapoints_from_class_to_oversample in datapoints_from_class_to_oversample_list:
        datapoints_to_oversample_list = datapoints_from_class_to_oversample.datapoints_to_oversample_list
        random_weights = []
        datapoints_and_neighbours = []
        print("oversampling " + str(datapoints_from_class_to_oversample.classes) + " " +
              str(datapoints_from_class_to_oversample.n_times_to_oversample) + " times")

        for datapoints_to_oversample in datapoints_to_oversample_list:
            taxonomy = datapoints_to_oversample.taxonomy
            if taxonomy == Taxonomy.SAFE:
                weight = safe_percentage
            elif taxonomy == Taxonomy.BORDERLINE:
                weight = borderline_percentage
            elif taxonomy == Taxonomy.RARE:
                weight = rare_percentage
            elif taxonomy == Taxonomy.OUTLIER:
                weight = 0
            else:
                exit("Taxonomy weight not supported " + taxonomy.value)

            random_weights.extend(np.full(len(datapoints_to_oversample.datapoints_and_neighbours), weight))
            datapoints_and_neighbours.extend(datapoints_to_oversample.datapoints_and_neighbours)

        if datapoints_and_neighbours and random_weights:
            for i in range(datapoints_from_class_to_oversample.n_times_to_oversample):
                # choose random
                random_datapoint_and_neighbour = random.choices(datapoints_and_neighbours, random_weights)[0]
                random_datapoint = random_datapoint_and_neighbour.datapoint
                neighbours = random_datapoint_and_neighbour.neighbours
                random_neighbour = random.choice(neighbours)
                new_synthetic_datapoint = create_synthetic_sample(dataset.features, random_datapoint, random_neighbour, neighbours)
                df = df.append(pd.Series(new_synthetic_datapoint), ignore_index=True)

    # save new dataset
    filename = dataset.get_oversampled_dataset_filename()
    save_dataset(df, filename)


def save_dataset(dataset: pd.DataFrame, filename: str):
    f = open(filename, "w+")
    f.write(dataset.to_csv(index=False))
    f.close()


def create_synthetic_sample(features: List[Feature], x1, x2, neighbours: List):
    synthetic_example = pd.Series()

    for feature in features:
        x1_value = x1[feature.name]
        x2_value = x2[feature.name]

        if feature.feature_type.__class__ == FeatureTypeContinuous:
            dif = x1_value - x2_value
            gap = np.random.random()
            synthetic_example_value = x1_value - gap * dif

        elif feature.feature_type.__class__ == FeatureTypeOrdinal:
            dif = x1_value - x2_value
            synthetic_example_value_float = x1_value - dif
            synthetic_example_value = int(synthetic_example_value_float)

        elif feature.feature_type.__class__ == FeatureTypeCategorical:
            datapoints = [x1]
            datapoints.extend(neighbours)
            synthetic_example_value = choose_most_common_value_from_all_datapoints(datapoints, feature.name) # TODO does most common??

        else:
            exit("Feature type not valid: " + feature.feature_type.__class__)

        synthetic_example[feature.name] = synthetic_example_value

    return synthetic_example


def choose_most_common_value_from_all_datapoints(datapoints, feature_name):
    counts = {}
    for datapoint in datapoints:
        value = datapoint[feature_name]
        if value in counts:
            counts[value] += 1
        else:
            counts[value] = 1

    most_common_value = max(counts.items(), key=operator.itemgetter(1))[0]

    return most_common_value


def get_datapoints_from_class_to_oversample_list(dataset: Dataset) -> List[DatapointsFromClassToOversample]:
    mappings = dataset.get_dataset_mappings()
    inversed_mappings = dataset.get_dataset_mappings_inverted()

    target_class = dataset.target_class.name
    positive_class = mappings[target_class][dataset.target_class.positive_class]
    negative_class = mappings[target_class][dataset.target_class.negative_class]
    df = dataset.get_train_dataset()
    datapoints_from_class_to_oversample_list = []
    unprivileged_classes_combs = get_combinations_sensitive_attributes(dataset.sensitive_classes, mappings)
    count_positive_privileged, count_negative_privileged = get_privileged_classes_counts(df, dataset.sensitive_classes,
                                                                                         target_class, positive_class,
                                                                                         negative_class, mappings)

    for comb in unprivileged_classes_combs:
        df_positive = df.copy(deep=True)
        df_negative = df.copy(deep=True)
        classes = {target_class: [dataset.target_class.positive_class]}

        for class_name, class_values in comb:
            df_positive = df_positive[(df_positive[class_name].isin(class_values)) & (df_positive[target_class] == positive_class)]
            df_negative = df_negative[(df_negative[class_name].isin(class_values)) & (df_negative[target_class] == negative_class)]
            classes[class_name] = [inversed_mappings[class_name][v] for v in class_values]

        count_positive_unprivileged = len(df_positive)
        count_negative_unprivileged = len(df_negative)

        desired_count_positive_unprivileged = count_negative_unprivileged * count_positive_privileged / count_negative_privileged
        n_times_to_oversample = int(desired_count_positive_unprivileged - count_positive_unprivileged)
        print("Difference in class " + str(classes) + " is " + str(n_times_to_oversample))

        effect = count_negative_unprivileged/count_positive_unprivileged - count_negative_privileged/count_positive_privileged
        print("Effect of class " + str(classes) + " is " + str(effect))

        datapoints_to_oversample_list = []

        for taxonomy in [Taxonomy.OUTLIER, Taxonomy.RARE, Taxonomy.BORDERLINE, Taxonomy.SAFE]:
            datapoints_and_neighbours = get_datapoints_and_neighbours_from_same_classes_and_taxonomy(dataset, df, classes, taxonomy)
            datapoints_to_oversample = DatapointsToOversample(taxonomy, datapoints_and_neighbours)
            datapoints_to_oversample_list.append(datapoints_to_oversample)

        datapoints_from_class_to_oversample = DatapointsFromClassToOversample(n_times_to_oversample,
                                                                              datapoints_to_oversample_list,
                                                                              classes)
        datapoints_from_class_to_oversample_list.append(datapoints_from_class_to_oversample)

    return datapoints_from_class_to_oversample_list


def get_privileged_classes_counts(df, sensitive_classes: List[SensitiveClass], target_class, positive_class,
                                  negative_class, mappings):
    df_positive = df.copy(deep=True)
    df_negative = df.copy(deep=True)

    for sensitive_class in sensitive_classes:
        class_name = sensitive_class.name
        class_values = [mappings[class_name][v] for v in sensitive_class.privileged_classes]
        df_positive = df_positive[(df_positive[class_name].isin(class_values)) & (df_positive[target_class] == positive_class)]
        df_negative = df_negative[(df_negative[class_name].isin(class_values)) & (df_negative[target_class] == negative_class)]

    return len(df_positive), len(df_negative)


def get_combinations_sensitive_attributes(sensitive_classes: List[SensitiveClass], mappings):
    classes_list = []

    for sensitive_class in sensitive_classes:
        classes = []
        class_name = sensitive_class.name
        priv_maps = [mappings[class_name][v] for v in sensitive_class.privileged_classes]
        classes.append((class_name, priv_maps))
        unpriv_maps = [mappings[class_name][v] for v in sensitive_class.unprivileged_classes]
        classes.append((class_name, unpriv_maps))
        classes_list.append(classes)

    return itertools.product(*classes_list)


def get_datapoints_and_neighbours_from_same_classes_and_taxonomy(dataset: Dataset,
                                                                 df: pd.DataFrame,
                                                                 classes: Dict,
                                                                 taxonomy: Taxonomy) -> List[DatapointAndNeighbours]:
    taxonomies = dataset.get_taxonomies_and_neighbours()
    indexes_of_datapoints_with_taxonomy = get_indexes_of_datapoints_with_taxonomy(taxonomies, taxonomy)
    df_subset = df.copy(deep=True)
    df_subset = df_subset.loc[indexes_of_datapoints_with_taxonomy]
    dataset_mappings = dataset.get_dataset_mappings()

    for class_name, class_values in dict(classes).items():
        class_values_mapped = [dataset_mappings[class_name][c] for c in class_values]
        df_subset = df_subset[df_subset[class_name].isin(class_values_mapped)]

    datapoints_and_neighbours = []
    datapoint_indexes = df_subset.index.to_list()

    for index in datapoint_indexes:
        neighbours = [df.iloc[index] for index in taxonomies[index].neighbours]
        datapoint = df.iloc[index]
        datapoint_and_neighbours = DatapointAndNeighbours(datapoint, neighbours)
        datapoints_and_neighbours.append(datapoint_and_neighbours)

    return datapoints_and_neighbours


def get_indexes_of_datapoints_with_taxonomy(taxonomies: List[TaxonomyAndNeighbours], taxonomy):
    indexes_of_datapoints_with_taxonomy = []

    for i in range(len(taxonomies)):
        if taxonomies[i].taxonomy == taxonomy:
            indexes_of_datapoints_with_taxonomy.append(i)

    return indexes_of_datapoints_with_taxonomy


def get_distributions_from_same_classes(dataset: Dataset, distributions: List[Distribution]) -> Dict:
    distributions_from_same_classes = {}

    for distribution in distributions:
        classes = {dataset.target_class.name: distribution.label.target_class_value}
        classes.update(distribution.label.sensitive_class_values)
        classes = frozenset(classes.items())

        if classes in distributions_from_same_classes:
            distributions_from_same_classes[classes].append(distribution)
        else:
            distributions_from_same_classes[classes] = [distribution]

    return distributions_from_same_classes
