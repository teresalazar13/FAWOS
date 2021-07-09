import os
import sys
import warnings

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import tree
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

    decision_tree = tree.DecisionTreeClassifier(random_state=dataset.seed)

    # https://medium.com/ai-in-plain-english/hyperparameter-tuning-of-decision-tree-classifier-using-gridsearchcv-2a6ebcaffeda
    param_grid = {'min_samples_split': np.linspace(0.1, 1.0, 5, endpoint=True),
                  'min_samples_leaf': np.linspace(0.1, 0.5, 5, endpoint=True),
                  'criterion': ['gini', 'entropy']
    }
    gd = GridSearchCV(estimator=decision_tree, param_grid=param_grid, verbose=True, error_score=0.0)

    gd.fit(X_train, y_train)
    pred_y = gd.predict(X_test)

    accuracy = round(accuracy_score(y_test, pred_y), 2)
    fairness_scores = fairness.get_fairness_results(dataset, X_test, pred_y)
    algorithm = Algorithm("Decision Trees", "#EF476F")

    return PerformanceResults(algorithm, accuracy, fairness_scores, gd.best_params_)
