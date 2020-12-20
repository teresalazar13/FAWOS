import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression

from models.dataset import Dataset
from classification import fairness
from classification.PerformanceResults import PerformanceResults
from classification.Algorithm import Algorithm


def classificate_and_evaluate(dataset: Dataset,
                              X_train: pd.DataFrame,
                              X_test: pd.DataFrame,
                              y_train: pd.Series,
                              y_test: pd.Series) -> PerformanceResults:
    lr = LogisticRegression(random_state=dataset.seed)
    lr.fit(X_train, y_train)
    pred_y = lr.predict(X_test)

    accuracy = round(accuracy_score(y_test, pred_y), 2)
    fairness_scores = fairness.get_fairness_results(dataset, X_test, pred_y)
    algorithm = Algorithm("Logistic Regression", "#06D6A0")

    return PerformanceResults(algorithm, accuracy, fairness_scores)
