import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

from code.models.dataset import Dataset


class FairnessMetrics:

    def __init__(self, sensitive_attribute_name, disparate_impact: float, adapted_disparate_impact: float,
                 average_abs_odds_difference: float, equal_opportunity_difference: float):
        self.sensitive_attribute_name = sensitive_attribute_name
        self.disparate_impact = disparate_impact
        self.adapted_disparate_impact = adapted_disparate_impact
        self.average_abs_odds_difference = average_abs_odds_difference
        self.equal_opportunity_difference = equal_opportunity_difference


def get_fairness_results(dataset: Dataset, X_test: pd.DataFrame, pred_y: pd.Series):
    binary_label_dataset, classified_dataset = create_fairness_datasets(dataset, X_test, pred_y)

    mappings = dataset.get_dataset_mappings()
    fairness_metrics_list = []
    adapted_disparate_impacts = []
    disparate_impacts = []
    average_abs_odds_differences = []
    equal_opportunity_differences = []

    for sensitive_class in dataset.sensitive_classes:
        class_name = sensitive_class.name
        class_values = [mappings[class_name][v] for v in sensitive_class.privileged_classes][0:1]
        privileged_groups = [{class_name: v} for v in class_values]
        class_values = [mappings[class_name][v] for v in sensitive_class.unprivileged_classes][0:1]
        unprivileged_groups = [{class_name: v} for v in class_values]

        # https://aif360.readthedocs.io/en/latest/modules/generated/aif360.metrics.ClassificationMetric.html#aif360.metrics.ClassificationMetric.disparate_impact
        classification_metric = ClassificationMetric(binary_label_dataset,
                                                     classified_dataset,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)

        disparate_impact = round(classification_metric.disparate_impact(), 2)
        disparate_impacts.append(disparate_impact)
        if disparate_impact <= 1:
            adapted_disparate_impact = disparate_impact
        else:
            adapted_disparate_impact = round(1 / disparate_impact, 2)
        adapted_disparate_impacts.append(adapted_disparate_impact)

        average_abs_odds_difference = classification_metric.average_abs_odds_difference()
        average_abs_odds_differences.append(average_abs_odds_difference)
        equal_opportunity_difference = classification_metric.equal_opportunity_difference()
        equal_opportunity_differences.append(equal_opportunity_difference)

        fairness_metrics = FairnessMetrics(class_name, disparate_impact, adapted_disparate_impact,
                                           average_abs_odds_difference, equal_opportunity_difference)
        fairness_metrics_list.append(fairness_metrics)

    adapted_disparate_impact_avg = round(sum(adapted_disparate_impacts) / len(adapted_disparate_impacts), 2)
    disparate_impact_avg = round(sum(disparate_impacts) / len(disparate_impacts), 2)
    average_abs_odds_difference_avg = round(sum(average_abs_odds_differences) / len(average_abs_odds_differences), 2)
    equal_opportunity_difference_avg = round(sum(equal_opportunity_differences) / len(equal_opportunity_differences), 2)

    fairness_metrics = FairnessMetrics("all", disparate_impact_avg, adapted_disparate_impact_avg,
                                       average_abs_odds_difference_avg, equal_opportunity_difference_avg)
    fairness_metrics_list.append(fairness_metrics)

    return fairness_metrics_list


def create_fairness_datasets(dataset: Dataset, X_test: pd.DataFrame, pred_y):
    test_dataset = dataset.get_test_dataset()
    label_names = [dataset.target_class.name]
    protected_attribute_names = [sensitive_class.name for sensitive_class in dataset.sensitive_classes]
    mappings = dataset.get_dataset_mappings()
    favorable_label = float(mappings[dataset.target_class.name][dataset.target_class.positive_class])
    unfavorable_label = float(mappings[dataset.target_class.name][dataset.target_class.negative_class])

    binary_label_dataset = BinaryLabelDataset(favorable_label=favorable_label, unfavorable_label=unfavorable_label,
                                              df=test_dataset, label_names=label_names, protected_attribute_names=protected_attribute_names)

    test_dataset_with_pred = X_test.copy(deep=True)
    test_dataset_with_pred[dataset.target_class.name] = pd.Series(pred_y)

    classified_dataset = BinaryLabelDataset(favorable_label=favorable_label, unfavorable_label=unfavorable_label,
                                            df=test_dataset_with_pred, label_names=label_names, protected_attribute_names=protected_attribute_names)

    return binary_label_dataset, classified_dataset
