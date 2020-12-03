import pandas as pd
from abc import ABCMeta, abstractmethod


class FeatureType(metaclass=ABCMeta):

    @abstractmethod
    def encode(self,
               dataset,
               feature_name: str,
               feature_values_raw_train: pd.Series,
               feature_values_raw_test: pd.Series) -> [pd.Series, pd.Series]:

        return feature_values_raw_train, feature_values_raw_test


    @abstractmethod
    def inverse_encode(self, dataset, feature_name: str, feature_value):

        return str(feature_value)
