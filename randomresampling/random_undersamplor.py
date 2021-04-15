import itertools
import numpy as np
import pandas as pd
from typing import List
import random

from models.dataset import Dataset
from models.SensitiveClass import SensitiveClass


class DatapointsToUndersampleRU:
    def __init__(self, n_times_to_undersample: int, df: pd.DataFrame) -> None:
        self.n_times_to_undersample = n_times_to_undersample
        self.df = df


def undersample(dataset: Dataset,
               datapoints_to_undersample: DatapointsToUndersampleRU):

    df = dataset.get_train_dataset()
    random.seed(dataset.seed)
    np.random.seed(dataset.seed)

    for i in range(datapoints_to_undersample.n_times_to_undersample):
        random_datapoint_to_remove = datapoints_to_undersample.df.sample()  # choose random
        df = df.drop(random_datapoint_to_remove)  # remove from dataset
        datapoints_to_undersample.df.drop(random_datapoint_to_remove)  # removes from list

    # save new dataset
    filename = dataset.get_random_undersampled_dataset_filename()
    save_dataset(df, filename)


def save_dataset(dataset: pd.DataFrame, filename: str):
    f = open(filename, "w+")
    f.write(dataset.to_csv(index=False))
    f.close()


def get_datapoints_from_class_to_undersample_list(dataset: Dataset) -> DatapointsToUndersampleRU:
    mappings = dataset.get_dataset_mappings()
    inversed_mappings = dataset.get_dataset_mappings_inverted()

    target_class = dataset.target_class.name
    positive_class = mappings[target_class][dataset.target_class.positive_class]
    negative_class = mappings[target_class][dataset.target_class.negative_class]
    df = dataset.get_train_dataset()
    unprivileged_classes_combs = get_combinations_sensitive_attributes(dataset.sensitive_classes, mappings)
    df_positive_priv, df_negative_priv, count_positive_privileged, count_negative_privileged = get_privileged_classes_counts(df, dataset.sensitive_classes,
                                                                                         target_class, positive_class,
                                                                                         negative_class, mappings)
    min_n_times_to_undersample = 10000

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

        desired_count_positive_privileged = count_positive_unprivileged * count_negative_privileged / count_negative_unprivileged
        n_times_to_undersample = int(count_positive_privileged - desired_count_positive_privileged)
        print("Difference in class " + str(classes) + " is " + str(n_times_to_undersample))

        effect = count_negative_unprivileged/count_positive_unprivileged - count_negative_privileged/count_positive_privileged
        print("Effect of class " + str(classes) + " is " + str(effect))

        if n_times_to_undersample < min_n_times_to_undersample:
            min_n_times_to_undersample = n_times_to_undersample

    return DatapointsToUndersampleRU(min_n_times_to_undersample, df_positive_priv)


def get_privileged_classes_counts(df, sensitive_classes: List[SensitiveClass], target_class, positive_class,
                                  negative_class, mappings):
    df_positive = df.copy(deep=True)
    df_negative = df.copy(deep=True)

    for sensitive_class in sensitive_classes:
        class_name = sensitive_class.name
        class_values = [mappings[class_name][v] for v in sensitive_class.privileged_classes]
        df_positive = df_positive[(df_positive[class_name].isin(class_values)) & (df_positive[target_class] == positive_class)]
        df_negative = df_negative[(df_negative[class_name].isin(class_values)) & (df_negative[target_class] == negative_class)]

    return df_positive, df_negative, len(df_positive), len(df_negative)


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
