import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
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

    knn = KNeighborsClassifier()

    # https://medium.com/@erikgreenj/k-neighbors-classifier-with-gridsearchcv-basics-3c445ddeb657
    param_grid = {
        "n_neighbors": [3, 5, 11, 19],
        "weights": ["uniform", "distance"],
        'metric': ["euclidean", "manhattan"]
    }
    gd = GridSearchCV(estimator=knn, param_grid=param_grid, verbose=True)

    gd.fit(X_train, y_train)
    pred_y = gd.predict(X_test)

    accuracy = round(accuracy_score(y_test, pred_y), 2)
    fairness_scores = fairness.get_fairness_results(dataset, X_test, pred_y)
    algorithm = Algorithm("KNN", "#564D4A")

    return PerformanceResults(algorithm, accuracy, fairness_scores, gd.best_params_)
