import os
import shutil
import pandas as pd
from typing import List, Dict
from abc import ABCMeta, abstractmethod

from models.Feature import Feature
from models.SensitiveClass import SensitiveClass
from models.TargetClass import TargetClass
from stats.Distribution import Distribution
from taxonomizing.TaxonomyAndNeighbours import TaxonomyAndNeighbours


class Dataset(metaclass=ABCMeta):

    def __init__(self,
                 name: str,
                 target_class: TargetClass,
                 sensitive_classes: List[SensitiveClass],
                 features: List[Feature],
                 test_size,
                 oversampling_factor: float):
        self.name = name
        self.target_class = target_class
        self.sensitive_classes = sensitive_classes
        self.features = features
        self.index = 1
        self.seed = self.index * 10
        self.test_size = test_size
        self.oversampling_factor = oversampling_factor

        if os.path.exists(self.get_sub_folder()):
            shutil.rmtree(self.get_sub_folder())

    def increase_index_and_seed(self):
        self.index += 1
        self.seed = self.index * 10

    def get_folder(self) -> str:
        return "./" + self.name + "/"

    def create_sub_directory(self):
        if not os.path.exists(self.get_sub_sub_folder()):
            os.makedirs(self.get_sub_sub_folder())

    def get_sub_folder(self) -> str:
        return self.get_folder() + "test-size-" + str(self.test_size) + "/oversampling-factor-" + str(self.oversampling_factor) + "/"

    def get_sub_sub_folder(self) -> str:
        return self.get_sub_folder() + "run-" + str(self.index) + "/"

    def get_raw_dataset_filename(self) -> str:
        return self.get_folder() + "raw_dataset.csv"

    def get_raw_dataset(self) -> pd.DataFrame:
        return pd.read_csv(self.get_raw_dataset_filename())

    def get_raw_transformed_dataset_filename(self) -> str:
        return self.get_sub_sub_folder() + "raw_transformed_dataset.csv"

    def get_raw_transformed_dataset(self) -> pd.DataFrame:
        return pd.read_csv(self.get_raw_transformed_dataset_filename())

    def get_raw_train_dataset_filename(self) -> str:
        return self.get_sub_sub_folder() + "raw_train_dataset.csv"

    def get_raw_train_dataset(self) -> pd.DataFrame:
        return pd.read_csv(self.get_raw_train_dataset_filename())

    def get_raw_test_dataset_filename(self) -> str:
        return self.get_sub_sub_folder() + "raw_test_dataset.csv"

    def get_raw_test_dataset(self) -> pd.DataFrame:
        return pd.read_csv(self.get_raw_test_dataset_filename())

    def get_train_dataset_filename(self) -> str:
        return self.get_sub_sub_folder() + "train_dataset.csv"

    def get_train_dataset(self) -> pd.DataFrame:
        return pd.read_csv(self.get_train_dataset_filename())

    def get_test_dataset_filename(self) -> str:
        return self.get_sub_sub_folder() + "test_dataset.csv"

    def get_test_dataset(self) -> pd.DataFrame:
        return pd.read_csv(self.get_test_dataset_filename())

    def get_oversampled_dataset_filename(self) -> str:
        return self.get_sub_sub_folder() + "oversampled_dataset.csv"

    def get_oversampled_dataset(self) -> pd.DataFrame:
        return pd.read_csv(self.get_oversampled_dataset_filename())

    def get_taxonomies_and_neighbours_filename(self) -> str:
        return self.get_sub_sub_folder() + "train_taxonomies_and_neighbours.txt"

    def get_taxonomies_and_neighbours(self) -> List[TaxonomyAndNeighbours]:
        return TaxonomyAndNeighbours.read_taxonomies_and_neighbours(self.get_taxonomies_and_neighbours_filename())

    def get_taxonomies_and_neighbours_oversampled_filename(self) -> str:
        return self.get_sub_sub_folder() + "oversampled_taxonomies_and_neighbours.txt"

    def get_taxonomies_and_neighbours_oversampled(self) -> List[TaxonomyAndNeighbours]:
        return TaxonomyAndNeighbours.read_taxonomies_and_neighbours(self.get_taxonomies_and_neighbours_oversampled_filename())

    def get_train_plot_filename(self) -> str:
        return self.get_sub_sub_folder() + "train_plot.png"

    def get_oversampled_plot_filename(self) -> str:
        return self.get_sub_sub_folder() + "oversampled_plot.png"

    def get_train_distributions_filename(self) -> str:
        return self.get_sub_sub_folder() + "train_distributions.txt"

    def get_train_distributions(self) -> List[Distribution]:
        return Distribution.read_distributions(self.get_train_distributions_filename())

    def get_oversampled_distributions_filename(self) -> str:
        return self.get_sub_sub_folder() + "oversampled_distributions.txt"

    def get_oversampled_distributions(self) -> List[Distribution]:
        return Distribution.read_distributions(self.get_oversampled_distributions_filename())

    def get_distributions_plot_filename(self):
        return self.get_sub_sub_folder() + "distributions_plot.png"

    def get_train_results_filename(self) -> str:
        return self.get_sub_sub_folder() + "train_results.txt"

    def get_oversampled_results_filename(self) -> str:
        return self.get_sub_sub_folder() + "oversampled_results.txt"

    def get_results_plot_filename(self) -> str:
        return self.get_sub_sub_folder() + "results_plot.png"

    def get_results_plot_overall_filename(self):
        return self.get_sub_folder() + "results_plot.png"

    def get_dataset_feature_value_mappings_filename(self) -> str:
        return self.get_sub_sub_folder() + "feature_value_mappings.txt"

    def reset_encoding_mapping(self):
        f = open(self.get_dataset_feature_value_mappings_filename(), "w+")
        f.close()

    def add_dataset_mapping_of_feature(self, feature_mappings: Dict, feature_name: str):
        f = open(self.get_dataset_feature_value_mappings_filename(), "a")

        for k, v in feature_mappings.items():
            f.write(feature_name + " " + str(k) + " " + str(v) + "\n")

        f.close()

    def get_dataset_mappings(self) -> Dict:
        return self.read_file_as_dict(self.get_dataset_feature_value_mappings_filename(), False)

    def get_dataset_mappings_inverted(self) -> Dict:
        return self.read_file_as_dict(self.get_dataset_feature_value_mappings_filename(), True)

    @staticmethod
    def read_file_as_dict(filename: str, inverted: bool) -> Dict:
        d = {}
        with open(filename) as f:
            for line in f:
                if not inverted:
                    (f, key, val) = line.split()
                    val = float(val)
                else:
                    (f, val, key) = line.split()
                    key = float(key)

                if f not in d:
                    d[f] = {}

                d[f][key] = val

        return d

    @abstractmethod
    def create_raw_transformed_dataset(self):  # default is just copy raw dataset
        raw_dataset = self.get_raw_dataset()
        raw_transformed_dataset_filename = self.get_raw_transformed_dataset_filename()
        f = open(raw_transformed_dataset_filename, "w+")
        f.write(raw_dataset.to_csv(index=False))
        f.close()
