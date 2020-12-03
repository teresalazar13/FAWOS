import pandas as pd

from models.FeatureType import FeatureType
from models.dataset import Dataset


class FeatureTypeContinuous(FeatureType):

    def encode(self,
               dataset: Dataset,
               feature_name: str,
               feature_values_train: pd.Series,
               feature_values_raw_test: pd.Series) -> [pd.Series, pd.Series]:

        return super().encode(dataset, feature_name, feature_values_train, feature_values_raw_test)


    def inverse_encode(self, dataset: Dataset, feature_name: str, feature_value):

        return super().inverse_encode(dataset, feature_value)
