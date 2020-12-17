import matplotlib.pyplot as plt
import numpy as np
from typing import List

from classification.Algorithm import Algorithm
from classification.fairness import FairnessMetrics


class PerformanceResults:

    def __init__(self,
                 algorithm: Algorithm,
                 accuracy: float,
                 fairness_metrics: FairnessMetrics):
        self.algorithm = algorithm
        self.accuracy = accuracy
        self.fairness_metrics = fairness_metrics


def save_performance_results_list(filename: str, performance_results_list: List[PerformanceResults]):
    f = open(filename, "w+")
    f.write("algorithm,accuracy,disparate_impact,metric\n")

    for results in performance_results_list:
        f.write(
            results.algorithm.name + ","
            + str(results.accuracy) + ","
            + str(results.fairness_metrics.disparate_impact) + ","
            + str(results.fairness_metrics.metric) + "\n"
        )

    f.close()


def create_results_plot(filename: str,
                        performance_results_train: List[PerformanceResults],
                        performance_results_oversampled: List[PerformanceResults]):

    plt.figure(figsize=(10, 10))
    legends = []

    for i in range(len(performance_results_train)):
        cv_score_train = performance_results_train[i].fairness_metrics.metric
        accuracy_train = performance_results_train[i].accuracy
        algorithm = performance_results_train[i].algorithm
        plt.scatter(cv_score_train, accuracy_train, c=algorithm.color)
        legend_train = algorithm.name + " baseline"
        if legend_train not in legends:
            legends.append(legend_train)

        cv_score_oversampled = performance_results_oversampled[i].fairness_metrics.metric
        accuracy_oversampled = performance_results_oversampled[i].accuracy
        plt.scatter(cv_score_oversampled, accuracy_oversampled, c=algorithm.color, marker="x")
        legend_oversampled = algorithm.name + " oversampled"
        if legend_oversampled not in legends:
            legends.append(legend_oversampled)

    axes = plt.gca()
    axes.set_xlim([-0.1, 1.1])
    axes.set_ylim([-0.1, 1.1])
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.title("Performance results of different classification algorithms")
    plt.xlabel("1 - |1 - Disparate Impact|")
    plt.ylabel("Accuracy")
    plt.legend(legends, loc='lower right')

    plt.savefig(filename)
