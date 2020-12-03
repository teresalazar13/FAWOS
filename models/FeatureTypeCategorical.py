import pandas as pd
from sklearn.preprocessing import LabelEncoder

from models.FeatureType import FeatureType
from models.dataset import Dataset


class FeatureTypeCategorical(FeatureType):

    def encode(self,
               dataset: Dataset,
               feature_name: str,
               feature_values_raw_train: pd.Series,
               feature_values_raw_test: pd.Series) -> [pd.Series, pd.Series]:

        all_feature_values = pd.concat([feature_values_raw_train, feature_values_raw_test])
        le = LabelEncoder()
        le.fit(all_feature_values)
        feature_values_train = le.transform(feature_values_raw_train)
        feature_values_test = le.transform(feature_values_raw_test)
        feature_values_train_series = pd.Series(feature_values_train, index=feature_values_raw_train.index)
        feature_values_test_series = pd.Series(feature_values_test, index=feature_values_raw_test.index)

        # Save mappings
        feature_mappings = dict(zip(feature_values_raw_train, feature_values_train))
        dataset.add_dataset_mapping_of_feature(feature_mappings, feature_name)

        return feature_values_train_series, feature_values_test_series


    def inverse_encode(self, dataset: Dataset, feature_name: str, feature_value):
        maps = dataset.get_dataset_mappings()

        return maps[feature_name][feature_value]
