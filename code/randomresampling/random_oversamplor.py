import itertools
import numpy as np
import pandas as pd
from typing import List
import random

from code.models.dataset import Dataset
from code.models.SensitiveClass import SensitiveClass


class DatapointsToOversampleRO:
    def __init__(self, n_times_to_oversample: int, df: pd.DataFrame) -> None:
        self.n_times_to_oversample = n_times_to_oversample
        self.df = df


def oversample(dataset: Dataset,
               datapoints_to_oversample_list: List[DatapointsToOversampleRO]):

    df = dataset.get_train_dataset()
    random.seed(dataset.seed)
    np.random.seed(dataset.seed)

    for datapoints_to_oversample in datapoints_to_oversample_list:
        for i in range(datapoints_to_oversample.n_times_to_oversample):
            random_datapoint_to_duplicate = datapoints_to_oversample.df.sample()  # choose random
            df = df.append(random_datapoint_to_duplicate, ignore_index=True)

    # save new dataset
    filename = dataset.get_random_oversampled_dataset_filename()
    save_dataset(df, filename)


def save_dataset(dataset: pd.DataFrame, filename: str):
    f = open(filename, "w+")
    f.write(dataset.to_csv(index=False))
    f.close()


def get_datapoints_from_class_to_oversample_list(dataset: Dataset) -> List[DatapointsToOversampleRO]:
    mappings = dataset.get_dataset_mappings()
    inversed_mappings = dataset.get_dataset_mappings_inverted()

    target_class = dataset.target_class.name
    positive_class = mappings[target_class][dataset.target_class.positive_class]
    negative_class = mappings[target_class][dataset.target_class.negative_class]
    df = dataset.get_train_dataset()
    unprivileged_classes_combs = get_combinations_sensitive_attributes(dataset.sensitive_classes, mappings)
    count_positive_privileged, count_negative_privileged = get_privileged_classes_counts(df, dataset.sensitive_classes,
                                                                                         target_class, positive_class,
                                                                                         negative_class, mappings)
    datapoints_to_oversample_list = []

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
        n_times_to_oversample = int((desired_count_positive_unprivileged - count_positive_unprivileged) * dataset.oversampling_factor)
        print("Difference in class " + str(classes) + " is " + str(n_times_to_oversample))

        effect = count_negative_unprivileged/count_positive_unprivileged - count_negative_privileged/count_positive_privileged
        print("Effect of class " + str(classes) + " is " + str(effect))

        datapoints_to_oversample = DatapointsToOversampleRO(n_times_to_oversample, df_positive)
        datapoints_to_oversample_list.append(datapoints_to_oversample)

    return datapoints_to_oversample_list


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
