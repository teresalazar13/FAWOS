import os
import sys
import warnings

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

from code.models.dataset import Dataset
from code.classification import fairness
from code.classification.PerformanceResults import PerformanceResults
from code.classification.Algorithm import Algorithm


def classificate_and_evaluate(dataset: Dataset,
                              X_train: pd.DataFrame,
                              X_test: pd.DataFrame,
                              y_train: pd.Series,
                              y_test: pd.Series) -> PerformanceResults:
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    gaussian_nb = GaussianNB()

    param_grid = {'var_smoothing': np.logspace(0, -9, num=100)}
    gd = GridSearchCV(estimator=gaussian_nb, param_grid=param_grid, verbose=True, error_score=0.0)

    gd.fit(X_train, y_train)
    pred_y = gd.predict(X_test)

    accuracy = round(accuracy_score(y_test, pred_y), 2)
    fairness_scores = fairness.get_fairness_results(dataset, X_test, pred_y)
    algorithm = Algorithm("Gaussian Naive Bayes", "#ffd166")

    return PerformanceResults(algorithm, accuracy, fairness_scores, gd.best_params_)
