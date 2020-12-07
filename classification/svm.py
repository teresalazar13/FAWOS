import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, log_loss

from models.dataset import Dataset
from classification import fairness
from classification.PerformanceResults import PerformanceResults
from classification.Algorithm import Algorithm


def classificate_and_evaluate(dataset: Dataset,
                              X_train: pd.DataFrame,
                              X_test: pd.DataFrame,
                              y_train: pd.Series,
                              y_test: pd.Series) -> PerformanceResults:
    svm = SVC(kernel='linear', C=1, probability=True)
    svm.fit(X_train, y_train)
    pred_y = svm.predict(X_test)

    accuracy = accuracy_score(y_test, pred_y)
    fairness_scores = fairness.get_fairness_results(dataset, X_test, pred_y)
    algorithm = Algorithm("SVM Linear", "#118AB2")

    return PerformanceResults(algorithm, accuracy, fairness_scores)
