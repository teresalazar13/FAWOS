import math
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

from models.dataset import Dataset


class FairnessMetrics:

    def __init__(self, sensitive_attribute_name, disparate_impact: float, adapted_disparate_impact: float):
        self.sensitive_attribute_name = sensitive_attribute_name
        self.disparate_impact = disparate_impact
        self.adapted_disparate_impact = adapted_disparate_impact


def get_fairness_results(dataset: Dataset, X_test: pd.DataFrame, pred_y: pd.Series):
    binary_label_dataset, classified_dataset = create_fairness_datasets(dataset, X_test, pred_y)

    mappings = dataset.get_dataset_mappings()
    fairness_metrics_list = []
    adapted_disparate_impacts = []
    disparate_impacts = []

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
        # disparate_impact = calculate_disparate_impact(dataset, X_test, pred_y)
        adapted_disparate_impact = round(1 - math.fabs(1 - disparate_impact), 2)
        adapted_disparate_impacts.append(adapted_disparate_impact)
        fairness_metrics = FairnessMetrics(class_name, disparate_impact, adapted_disparate_impact)
        fairness_metrics_list.append(fairness_metrics)

    adapted_disparate_impact_avg = round(sum(adapted_disparate_impacts) / len(adapted_disparate_impacts), 2)
    disparate_impact_avg = round(sum(disparate_impacts) / len(disparate_impacts), 2)
    fairness_metrics = FairnessMetrics("all", disparate_impact_avg, adapted_disparate_impact_avg)
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


"""
def calculate_disparate_impact(dataset, X_test, pred_y):
    df = X_test.copy(deep=True)
    df[dataset.target_class.name] = pd.Series(pred_y)
    print(df.head(10))
    target_class = dataset.target_class.name
    sensitive_class = dataset.sensitive_classes[0].name

    mappings = dataset.get_dataset_mappings()
    positive_class = mappings[target_class][dataset.target_class.positive_class]
    negative_class = mappings[target_class][dataset.target_class.negative_class]
    unprivileged_classes = [mappings[sensitive_class][s] for s in dataset.sensitive_classes[0].unprivileged_classes]
    privileged_classes = [mappings[sensitive_class][s] for s in dataset.sensitive_classes[0].privileged_classes]

    pp = len(df[(df[target_class] == positive_class) & (df[sensitive_class].isin(privileged_classes))])
    pn = len(df[(df[target_class] == negative_class) & (df[sensitive_class].isin(privileged_classes))])
    up = len(df[(df[target_class] == positive_class) & (df[sensitive_class].isin(unprivileged_classes))])
    un = len(df[(df[target_class] == negative_class) & (df[sensitive_class].isin(unprivileged_classes))])

    nominator = up / (un + up)
    denominator = pp / (pp + pn)

    return nominator / denominator"""