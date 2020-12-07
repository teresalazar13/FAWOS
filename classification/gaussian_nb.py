import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.naive_bayes import GaussianNB

from models.dataset import Dataset
from classification import fairness
from classification.PerformanceResults import PerformanceResults
from classification.Algorithm import Algorithm


def classificate_and_evaluate(dataset: Dataset,
                              X_train: pd.DataFrame,
                              X_test: pd.DataFrame,
                              y_train: pd.Series,
                              y_test: pd.Series) -> PerformanceResults:
    gaussian_nb = GaussianNB()
    gaussian_nb.fit(X_train, y_train)
    pred_y = gaussian_nb.predict(X_test)

    accuracy = accuracy_score(y_test, pred_y)
    fairness_scores = fairness.get_fairness_results(dataset, X_test, pred_y)
    algorithm = Algorithm("Gaussian Naive Bayes", "#ffd166")

    return PerformanceResults(algorithm, accuracy, fairness_scores)
