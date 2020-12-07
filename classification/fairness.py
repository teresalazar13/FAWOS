import math
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

from models.dataset import Dataset


class FairnessMetrics:

    def __init__(self, disparate_impact: float, metric: float):
        self.disparate_impact = disparate_impact
        self.metric = metric


def get_fairness_results(dataset: Dataset, X_test: pd.DataFrame, pred_y: pd.Series):
    binary_label_dataset, classified_dataset = create_fairness_datasets(dataset, X_test, pred_y)
    privileged_groups, unprivileged_groups = get_priviledged_and_unprivileged_groups(dataset)

    # https://aif360.readthedocs.io/en/latest/modules/generated/aif360.metrics.ClassificationMetric.html#aif360.metrics.ClassificationMetric.disparate_impact
    classification_metric = ClassificationMetric(binary_label_dataset,
                                                 classified_dataset,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)

    disparate_impact = classification_metric.disparate_impact()
    # disparate_impact = calculate_disparate_impact(dataset, X_test, pred_y)
    my_metric = 1 - math.fabs(1 - disparate_impact)

    return FairnessMetrics(disparate_impact, my_metric)


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


def get_priviledged_and_unprivileged_groups(dataset: Dataset):
    privileged_groups = []
    unprivileged_groups = []
    dataset_mappings = dataset.get_dataset_mappings()

    for feature in dataset.features:
        for sensitive_class in dataset.sensitive_classes:
            if feature.name == sensitive_class.name:
                for priv in sensitive_class.privileged_classes:
                    v_priv = dataset_mappings[feature.name][priv]
                    privileged_groups.append({sensitive_class.name: v_priv})
                for unpriv in sensitive_class.unprivileged_classes:
                    v_unpriv = dataset_mappings[feature.name][unpriv]
                    unprivileged_groups.append({sensitive_class.name: v_unpriv})

    return privileged_groups, unprivileged_groups


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