import os
import sys
import warnings

import pandas as pd
import numpy as np
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from models.dataset import Dataset
from classification import fairness
from classification.PerformanceResults import PerformanceResults
from classification.Algorithm import Algorithm


def classificate_and_evaluate(dataset: Dataset,
                              X_train: pd.DataFrame,
                              X_test: pd.DataFrame,
                              y_train: pd.Series,
                              y_test: pd.Series) -> PerformanceResults:
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    lr = LogisticRegression(random_state=dataset.seed)

    # https://www.kaggle.com/enespolat/grid-search-with-logistic-regression
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html


    param_grid = {
        "C": np.logspace(-4, 4, 20),
        "penalty": ["l1", "l2"],
        'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
    }
    gd = GridSearchCV(estimator=lr, param_grid=param_grid, verbose=True, error_score=0.0)

    gd.fit(X_train, y_train)
    pred_y = gd.predict(X_test)

    accuracy = round(accuracy_score(y_test, pred_y), 2)
    fairness_scores = fairness.get_fairness_results(dataset, X_test, pred_y)
    algorithm = Algorithm("Logistic Regression", "#06D6A0")

    return PerformanceResults(algorithm, accuracy, fairness_scores, gd.best_params_)
