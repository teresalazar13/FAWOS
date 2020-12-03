from typing import List
import pandas as pd

from models.FeatureType import FeatureType
from models.dataset import Dataset


class FeatureTypeOrdinal(FeatureType):

    def __init__(self, order: List):
        self.order = order


    def encode(self,
               dataset: Dataset,
               feature_name: str,
               feature_values_raw_train: pd.Series,
               feature_values_raw_test: pd.Series) -> [pd.Series, pd.Series]:

        feature_values_train = feature_values_raw_train[:].apply(lambda x: self.order.index(x))
        feature_values_test = feature_values_raw_test[:].apply(lambda x: self.order.index(x))

        # Save mappings
        feature_mappings = dict(zip(feature_values_raw_train, feature_values_train))
        dataset.add_dataset_mapping_of_feature(feature_mappings, feature_name)

        return feature_values_train, feature_values_test


    def inverse_encode(self, dataset: Dataset, feature_name: str, feature_value):
        maps = dataset.get_dataset_mappings()

        return maps[feature_name][feature_value]
