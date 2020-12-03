import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn import tree

from models.dataset import Dataset
from classification import fairness


def classificate_and_evaluate(dataset: Dataset,
                              X_train: pd.DataFrame,
                              X_test: pd.DataFrame,
                              y_train: pd.Series,
                              y_test: pd.Series):
    decision_tree = tree.DecisionTreeClassifier()
    decision_tree.fit(X_train, y_train)
    pred_y = decision_tree.predict(X_test)
    pp = decision_tree.predict_proba(X_test)

    results_string = "Decision Tree\n"
    results_string += "Log Loss: {:.5f}\n".format(log_loss(y_test, pp))
    results_string += "Classification accuracy: {:.5f}\n".format(accuracy_score(y_test, pred_y))
    results_string += fairness.get_fairness_results(dataset, X_test, pred_y)

    return results_string
