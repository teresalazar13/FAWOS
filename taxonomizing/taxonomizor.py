from typing import List
import numpy as np
import pandas as pd

from models.FeatureTypeCategorical import FeatureTypeCategorical
from models.FeatureTypeContinuous import FeatureTypeContinuous
from models.FeatureTypeOrdinal import FeatureTypeOrdinal
from models.dataset import Dataset
from taxonomizing.TaxonomyAndNeighbours import TaxonomyAndNeighbours
from taxonomizing.Taxonomy import Taxonomy


def create_taxonomies_and_neighbours(dataset: Dataset,
                                     X_train: pd.DataFrame,
                                     y_train: pd.Series,
                                     taxonomies_filename: str):

    distances = heomDist(dataset, X_train)
    taxonomies_and_neighbours = determine_taxonomies_and_neighbours(distances, dataset, X_train, y_train)
    TaxonomyAndNeighbours.save_taxonomies_and_neighbours(taxonomies_filename, taxonomies_and_neighbours)


def determine_taxonomies_and_neighbours(distances,
                                        dataset: Dataset,
                                        X_train: pd.DataFrame,
                                        y_train: pd.Series) -> List[TaxonomyAndNeighbours]:
    taxonomies_and_neighbours = []
    neighbours_list = calculate_neighbours(distances, dataset, X_train, y_train)

    for i in range(len(neighbours_list)):
        count = len(neighbours_list[i])
        taxonomy = ""

        if count == 0:
            taxonomy = Taxonomy.OUTLIER

        elif count == 1:
            solo_neighbour = neighbours_list[i][0]
            neighbours_of_solo_neighbour = neighbours_list[solo_neighbour]

            if len(neighbours_of_solo_neighbour) == 0 or (len(neighbours_of_solo_neighbour) == 1 and neighbours_of_solo_neighbour[0] == i):
                taxonomy = Taxonomy.RARE

            else:
                taxonomy = Taxonomy.BORDERLINE

        elif count in [2, 3]:
            taxonomy = Taxonomy.BORDERLINE

        elif count in [4, 5]:
            taxonomy = Taxonomy.SAFE

        taxonomy_and_neighbours = TaxonomyAndNeighbours(taxonomy, neighbours_list[i])
        taxonomies_and_neighbours.append(taxonomy_and_neighbours)

    return taxonomies_and_neighbours


def calculate_neighbours(distances, dataset: Dataset, X_train: pd.DataFrame, y_train: pd.Series):
    neighbours_list = []

    for i in range(len(distances)):
        datapoint = X_train.iloc[i]
        target_datapoint = y_train.iloc[i]
        neighbours_indexes = np.array(distances[i]).argsort()[:5]
        neighbours = []

        for neighbour in neighbours_indexes:
            neighbour_datapoint = X_train.iloc[neighbour, :]
            target_neighbour_datapoint = y_train.iloc[neighbour]

            points_belong_to_same_sensitive_classes = True
            for sensitive_class in dataset.sensitive_classes:
                if datapoint[sensitive_class.name] != neighbour_datapoint[sensitive_class.name]:
                    points_belong_to_same_sensitive_classes = False
                    break

            if points_belong_to_same_sensitive_classes and target_datapoint == target_neighbour_datapoint:
                neighbours.append(neighbour)

        neighbours_list.append(neighbours)

    return neighbours_list


def heomDist(dataset: Dataset, X_train: pd.DataFrame):
    val_max_col = []
    val_min_col = []

    for col in X_train.columns:
        max = X_train[col].max()
        val_max_col.append(max)
        min = X_train[col].min()
        val_min_col.append(min)

    N = X_train.shape[0]
    distances = np.full((N, N,), np.inf)

    for i in range(N):
        if i % 10 == 0:
            print("Calculating distances of point:", i)
        for j in range(i + 1, N):
            distance = heom(dataset, X_train, val_max_col, val_min_col, i, j)
            distances[i][j] = distance
            distances[j][i] = distance

    return distances


def heom(dataset: Dataset, data, val_max_col, val_min_col, m, n):
    """This function computes Heterogeneous Euclidean Overlap Metric distance
    between m-th sample and n-th sample in a given dataset
    Parameters
    ----------
    data: array, shape(n_instances,n_features)
        array containing the original dataset

    val_max_col: list of floats
        max value for each column
    val_min_col: list of floats
        min value for each column

    m: int
        row number of the first sample in the dataset
    n: int
        row number of the second sample in the dataset
    Returns
    -------
    dist_heom: float
        HEOM distance between i-th sample and k-th sample in dataset specified
        by data
    """
    dist_sum = 0
    i = 0
    for feature in dataset.features[:-1]:
        mm = data[feature.name].iloc[m]
        nn = data[feature.name].iloc[n]

        dist_temp = 0
        if mm == '' or nn == '':
            dist_temp = 1

        # 'binary', 'categorical', 'ordinal'
        elif isinstance(feature.feature_type, (FeatureTypeCategorical, FeatureTypeOrdinal)):
            if mm == nn:
                dist_temp = 0
            else:
                dist_temp = 1

        # 'interval', 'continuous'
        elif isinstance(feature.feature_type, FeatureTypeContinuous):
            if val_max_col[i] - val_min_col[i] == 0:
                dist_temp = 0
            else:
                dist_temp = (float(mm) - float(nn)) / (val_max_col[i] - val_min_col[i])

        else:
            print("Taxonomize Exception - " + mm + " or " + nn + "values not recognized")
            exit()

        dist_sum += dist_temp ** 2
        i += 1

    dist_heom = dist_sum ** 0.5

    return dist_heom
