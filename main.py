import pandas as pd
import argparse
import sys
import classification.svm as svm
import classification.gaussian_nb as gaussian_nb
import classification.decision_tree as decision_tree
import classification.knn as knn
import classification.logistic_regression as logistic_regression
import classification.PerformanceResults as performance_results
from typing import List

from models.Adult import Adult
from models.Credit import Credit
from models.Ricci import Ricci
from models.dataset import Dataset
from oversampling import oversamplor
from preprocessing import preprocessor
from taxonomizing import taxonomizor
from stats import stats_and_plots_creator
from taxonomizing.TaxonomyAndNeighbours import TaxonomyAndNeighbours
from randomresampling import random_oversamplor
from randomresampling import random_undersamplor


def get_dataset(dataset_name: str,
                test_size: float,
                oversampling_factor: float,
                safe_weight: float,
                borderline_weight: float,
                rare_weight: float) -> Dataset:
    if dataset_name == "adult":
        return Adult(test_size, oversampling_factor, safe_weight, borderline_weight, rare_weight)
    elif dataset_name == "credit":
        return Credit(test_size, oversampling_factor, safe_weight, borderline_weight, rare_weight)
    elif dataset_name == "ricci":
        return Ricci(test_size, oversampling_factor, safe_weight, borderline_weight, rare_weight)


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
                         logistic_regression.classificate_and_evaluate(dataset, X_train, X_test, y_train, y_test),
                         knn.classificate_and_evaluate(dataset, X_train, X_test, y_train, y_test)]

    performance_results.save_performance_results_list(results_filename, perf_results_list)

    return perf_results_list


def create_stats(dataset: Dataset,
                 df: pd.DataFrame,
                 taxonomies_and_neighbours: List[TaxonomyAndNeighbours],
                 stats_filename: str) -> list:

    return stats_and_plots_creator.save_stats(dataset, df, taxonomies_and_neighbours, stats_filename)


def oversample(dataset: Dataset, safe_weight, borderline_weight, rare_weight):
    datapoints_from_class_to_oversample_list = oversamplor.get_datapoints_from_class_to_oversample_list(dataset)
    oversamplor.oversample(dataset, datapoints_from_class_to_oversample_list, safe_weight, borderline_weight, rare_weight)


def random_oversample(dataset: Dataset):
    datapoints_from_class_to_oversample_list = random_oversamplor.get_datapoints_from_class_to_oversample_list(dataset)
    random_oversamplor.oversample(dataset, datapoints_from_class_to_oversample_list)


def random_undersample(dataset: Dataset):
    datapoints_from_class_to_undersample = random_undersamplor.get_datapoints_from_class_to_undersample_list(dataset)
    random_undersamplor.undersample(dataset, datapoints_from_class_to_undersample)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=["credit", "adult", "ricci"], required=True, help='dataset name')
    parser.add_argument('--test_size', choices=["0.2", "0.3"], required=True, help='test size')
    parser.add_argument('--oversampling_factor', required=True, help='oversampling factor')
    parser.add_argument('--taxonomy_weights', required=True, help='taxonomy_weights', nargs='+')
    parser.add_argument('--n_runs', choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], required=True,
                        help='n_runs')
    args = parser.parse_args(sys.argv[1:])
    safe_weight, borderline_weight, rare_weight = [float(w) for w in args.taxonomy_weights]

    dataset = get_dataset(args.dataset, float(args.test_size), float(args.oversampling_factor),
                          safe_weight, borderline_weight, rare_weight)
    performance_results_train_list = []
    performance_results_oversampled_list = []

    for i in range(int(args.n_runs)):
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
        oversample(dataset, safe_weight, borderline_weight, rare_weight)

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
        # stats_and_plots_creator.create_points_plot(str_labels_oversampled, oversampled_dataset, dataset.get_train_plot_filename(), dataset.get_oversampled_plot_filename(), len(train_dataset))

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

        # COMPARISON WITH RANDOM OVERSAMPLOR
        random_oversample(dataset)

        random_oversampled_dataset = dataset.get_random_oversampled_dataset()
        X_train_random_oversampled = random_oversampled_dataset.loc[:, random_oversampled_dataset.columns != dataset.target_class.name]
        y_train_random_oversampled = random_oversampled_dataset[dataset.target_class.name]
        results_filename_random_oversampled = dataset.get_random_oversampled_results_filename()

        test_dataset = dataset.get_test_dataset()
        X_test = test_dataset.loc[:, test_dataset.columns != dataset.target_class.name]
        y_test = test_dataset[dataset.target_class.name]
        classify_and_evaluate(dataset, X_train_random_oversampled, y_train_random_oversampled,
                            X_test, y_test, results_filename_random_oversampled)

        # COMPARISON WITH RANDOM UNDERSAMPLOR
        random_undersample(dataset)

        random_undersampled_dataset = dataset.get_random_undersampled_dataset()
        X_train_random_undersampled = random_undersampled_dataset.loc[:,
                                     random_undersampled_dataset.columns != dataset.target_class.name]
        y_train_random_undersampled = random_undersampled_dataset[dataset.target_class.name]
        results_filename_random_undersampled = dataset.get_random_undersampled_results_filename()

        classify_and_evaluate(dataset, X_train_random_undersampled, y_train_random_undersampled,
                                X_test, y_test, results_filename_random_undersampled)

        dataset.increase_index_and_seed()

    """
    # Comparison
    # Plot Performance Results
    results_filename = dataset.get_results_plot_overall_filename()
    performance_results.create_results_plot(results_filename, performance_results_train_list, performance_results_oversampled_list)
    number_algorithms = 5
    for i in range(number_algorithms):
        performance_results_alg_train = [performance_results_train_list[j + i] for j in range(0, len(performance_results_train_list), number_algorithms)]
        performance_results_alg_oversampled = [performance_results_oversampled_list[j + i] for j in range(0, len(performance_results_oversampled_list), number_algorithms)]
        results_filename_alg = results_filename[:-4] + "_" + performance_results_train_list[i].algorithm.name.replace(" ", "_").lower() + ".png"
        performance_results.create_results_plot(results_filename_alg, performance_results_alg_train, performance_results_alg_oversampled)
    """
