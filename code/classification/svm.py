import os
import sys
import warnings

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from code.models import dataset
from code.classification import fairness
from code.classification.PerformanceResults import PerformanceResults
from code.classification.Algorithm import Algorithm


def classificate_and_evaluate(dataset: dataset.Dataset,
                              X_train: pd.DataFrame,
                              X_test: pd.DataFrame,
                              y_train: pd.Series,
                              y_test: pd.Series) -> PerformanceResults:
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    svm = SVC(kernel='linear', C=1, probability=True, random_state=dataset.seed)

    # https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ["scale", "auto"],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    gd = GridSearchCV(estimator=svm, param_grid=param_grid, verbose=True, error_score=0.0)

    gd.fit(X_train, y_train)
    pred_y = gd.predict(X_test)

    accuracy = round(accuracy_score(y_test, pred_y), 2)
    fairness_scores = fairness.get_fairness_results(dataset, X_test, pred_y)
    algorithm = Algorithm("SVM Linear", "#118AB2")

    return PerformanceResults(algorithm, accuracy, fairness_scores, gd.best_params_)
