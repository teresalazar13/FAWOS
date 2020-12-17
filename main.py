from typing import List

import pandas as pd

import classification.svm as svm
import classification.gaussian_nb as gaussian_nb
import classification.decision_tree as decision_tree
import classification.logistic_regression as logistic_regression
import classification.PerformanceResults as performance_results
from models.Adult import Adult
from models.Credit import Credit
from models.dataset import Dataset
from oversampling import oversamplor
from preprocessing import preprocessor
from taxonomizing import taxonomizor
from stats import stats_and_plots_creator
from taxonomizing.TaxonomyAndNeighbours import TaxonomyAndNeighbours


def get_dataset() -> Dataset:
    #return Adult()
    return Credit()


def preprocess(dataset: Dataset):
    preprocessor.apply_specific_dataset_processing(dataset)
    X_train, X_test, y_train, y_test = preprocessor.train_test_split_and_save(dataset)
    preprocessor.create_preprocessed_train_and_test_datasets(dataset, X_train, X_test, y_train, y_test)


def taxonomize(dataset: Dataset,
               X_train: pd.DataFrame,
               y_train: pd.Series,
               taxonomies_filename: str):

    taxonomizor.create_taxonomies_and_neighbours(dataset, X_train, y_train, taxonomies_filename)


def classify_and_evaluate(dataset: Dataset,
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_test: pd.DataFrame,
                          y_test: pd.Series,
                          results_filename: str) -> List[performance_results.PerformanceResults]:
    perf_results_list = [svm.classificate_and_evaluate(dataset, X_train, X_test, y_train, y_test),
                         gaussian_nb.classificate_and_evaluate(dataset, X_train, X_test, y_train, y_test),
                         decision_tree.classificate_and_evaluate(dataset, X_train, X_test, y_train, y_test),
                         logistic_regression.classificate_and_evaluate(dataset, X_train, X_test, y_train, y_test)]

    performance_results.save_performance_results_list(results_filename, perf_results_list)

    return perf_results_list


def create_stats(dataset: Dataset,
                 df: pd.DataFrame,
                 taxonomies_and_neighbours: List[TaxonomyAndNeighbours],
                 stats_filename: str) -> list:

    return stats_and_plots_creator.save_stats(dataset, df, taxonomies_and_neighbours, stats_filename)


def oversample(dataset: Dataset):
    datapoints_from_class_to_oversample_list = oversamplor.get_datapoints_from_class_to_oversample_list(dataset)
    oversamplor.oversample(dataset, datapoints_from_class_to_oversample_list, 0, 0.4, 0.6)


if __name__ == '__main__':
    dataset = get_dataset()
    performance_results_train_list = []
    performance_results_oversampled_list = []

    for i in range(5):
        dataset.create_sub_directory()

        # Preprocess
        preprocess(dataset)

        train_dataset = dataset.get_train_dataset()
        X_train = train_dataset.loc[:, train_dataset.columns != dataset.target_class.name]
        y_train = train_dataset[dataset.target_class.name]
        test_dataset = dataset.get_test_dataset()
        X_test = test_dataset.loc[:, test_dataset.columns != dataset.target_class.name]
        y_test = test_dataset[dataset.target_class.name]
        # Classificate + Evaluate Train
        results_filename_train = dataset.get_train_results_filename()
        performance_results_train = classify_and_evaluate(dataset, X_train, y_train, X_test, y_test, results_filename_train)

        # Taxonomize Train
        taxonomies_filename = dataset.get_taxonomies_and_neighbours_filename()
        taxonomize(dataset, X_train, y_train, taxonomies_filename)

        # Create Stats Train
        _ = create_stats(dataset,
                         train_dataset,
                         dataset.get_taxonomies_and_neighbours(),
                         dataset.get_train_distributions_filename())

        # Oversample
        oversample(dataset)

        # Taxonomize Oversampled
        oversampled_dataset = dataset.get_oversampled_dataset()
        X_train_oversampled = oversampled_dataset.loc[:, oversampled_dataset.columns != dataset.target_class.name]
        y_train_oversampled = oversampled_dataset[dataset.target_class.name]
        taxonomies_filename_oversampled = dataset.get_taxonomies_and_neighbours_oversampled_filename()
        taxonomize(dataset, X_train_oversampled, y_train_oversampled, taxonomies_filename_oversampled)

        # Create Stats Oversampled
        str_labels_oversampled = create_stats(dataset,
                                              oversampled_dataset,
                                              dataset.get_taxonomies_and_neighbours_oversampled(),
                                              dataset.get_oversampled_distributions_filename())

        # Create Plots Train and Oversampled
        stats_and_plots_creator.create_points_plot(str_labels_oversampled,
                                                   oversampled_dataset,
                                                   dataset.get_train_plot_filename(),
                                                   dataset.get_oversampled_plot_filename(),
                                                   len(train_dataset))

        # Classificate + Evaluate Oversampled
        results_filename_oversampled = dataset.get_oversampled_results_filename()
        performance_results_oversampled = classify_and_evaluate(dataset, X_train_oversampled, y_train_oversampled, X_test, y_test, results_filename_oversampled)

        # Comparison
        # Plot Distributions
        stats_and_plots_creator.create_distributions_plot(dataset)
        # Plot Performance Results
        filename = dataset.get_results_plot_filename()
        performance_results.create_results_plot(filename, performance_results_train, performance_results_oversampled)
        performance_results_train_list.extend(performance_results_train)
        performance_results_oversampled_list.extend(performance_results_oversampled)

        dataset.increase_index()

    # Comparison
    # Plot Performance Results
    filename = dataset.get_results_plot_overall_filename()
    performance_results.create_results_plot(filename, performance_results_train_list, performance_results_oversampled_list)
