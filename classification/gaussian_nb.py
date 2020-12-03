import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.naive_bayes import GaussianNB

from models.dataset import Dataset
from classification import fairness


def classificate_and_evaluate(dataset: Dataset,
                              X_train: pd.DataFrame,
                              X_test: pd.DataFrame,
                              y_train: pd.Series,
                              y_test: pd.Series):
    gaussian_nb = GaussianNB()
    gaussian_nb.fit(X_train, y_train)
    pred_y = gaussian_nb.predict(X_test)
    pp = gaussian_nb.predict_proba(X_test)

    results_string = "Gaussian Naive Bayes\n"
    results_string += "Log Loss: {:.5f}\n".format(log_loss(y_test, pp))
    results_string += "Classification accuracy: {:.5f}\n".format(accuracy_score(y_test, pred_y))
    results_string += fairness.get_fairness_results(dataset, X_test, pred_y)

    return results_string
