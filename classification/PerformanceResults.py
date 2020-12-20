import matplotlib.pyplot as plt
import numpy as np
from typing import List

from classification.Algorithm import Algorithm
from classification.fairness import FairnessMetrics


class PerformanceResults:

    def __init__(self,
                 algorithm: Algorithm,
                 accuracy: float,
                 fairness_metrics_list: List[FairnessMetrics]):
        self.algorithm = algorithm
        self.accuracy = accuracy
        self.fairness_metrics_list = fairness_metrics_list


def save_performance_results_list(filename: str, performance_results_list: List[PerformanceResults]):
    f = open(filename, "w+")

    fairness_metrics_names = ""
    for fairness_metrics in performance_results_list[0].fairness_metrics_list:
        fairness_metrics_names += fairness_metrics.sensitive_attribute_name + "_disparate_impact," + fairness_metrics.sensitive_attribute_name + "_adapted_disparate_impact,"
    f.write("algorithm,accuracy," + fairness_metrics_names[:-1] + "\n")

    for results in performance_results_list:
        f.write(
            results.algorithm.name + ","
            + str(results.accuracy) + ","
        )
        for fairness_metrics in results.fairness_metrics_list:
            f.write(
                str(fairness_metrics.disparate_impact) + ","
                + str(fairness_metrics.adapted_disparate_impact) + ","
            )
        f.write("\n")
    f.close()


def create_results_plot(filename: str,
                        performance_results_train: List[PerformanceResults],
                        performance_results_oversampled: List[PerformanceResults]):

    for i in range(len(performance_results_train[0].fairness_metrics_list)):
        plt.figure(figsize=(10, 10))
        legends = []

        for j in range(len(performance_results_train)):
            cv_score_train = performance_results_train[j].fairness_metrics_list[i].adapted_disparate_impact
            accuracy_train = performance_results_train[j].accuracy
            algorithm = performance_results_train[j].algorithm
            plt.scatter(cv_score_train, accuracy_train, c=algorithm.color)
            legend_train = algorithm.name + " baseline"
            if legend_train not in legends:
                legends.append(legend_train)

            cv_score_oversampled = performance_results_oversampled[j].fairness_metrics_list[i].adapted_disparate_impact
            accuracy_oversampled = performance_results_oversampled[j].accuracy
            plt.scatter(cv_score_oversampled, accuracy_oversampled, c=algorithm.color, marker="x")
            legend_oversampled = algorithm.name + " oversampled"
            if legend_oversampled not in legends:
                legends.append(legend_oversampled)

        axes = plt.gca()
        axes.set_xlim([-0.1, 1.1])
        axes.set_ylim([-0.1, 1.1])
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        attribute_name = performance_results_train[j].fairness_metrics_list[i].sensitive_attribute_name
        if attribute_name == "all":
            plt.title("Performance results of different classification algorithms for all attributes")
        else:
            plt.title("Performance results of different classification algorithms for " + attribute_name + " attribute")
        plt.xlabel("1 - |1 - Disparate Impact|")
        plt.ylabel("Accuracy")
        plt.legend(legends, loc='lower right')

        plt.savefig(filename[:-4] + "_" + attribute_name + ".png")
