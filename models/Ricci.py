from models.Feature import Feature
from models.FeatureTypeCategorical import FeatureTypeCategorical
from models.FeatureTypeContinuous import FeatureTypeContinuous
from models.SensitiveClass import SensitiveClass
from models.TargetClass import TargetClass
from models.dataset import Dataset


class Ricci(Dataset):

    def __init__(self, test_size, oversampling_factor):
        name = "ricci"
        target_class = TargetClass("Combine", "Positive", "Negative")
        sensitive_class_race = SensitiveClass("Race", ["W"], ["H", "B"])
        sensitive_classes = [sensitive_class_race]
        features = [
            self.get_position(),
            self.get_oral(),
            self.get_written(),
            self.get_race(),
            self.get_combine(),
        ]

        super().__init__(name, target_class, sensitive_classes, features, test_size, oversampling_factor)

    def create_raw_transformed_dataset(self):
        raw_dataset = self.get_raw_dataset()

        positive = raw_dataset['Combine'] >= 70.0
        raw_dataset.loc[positive, 'Combine'] = "Positive"
        negative = raw_dataset['Combine'] != "Positive"
        raw_dataset.loc[negative, 'Combine'] = "Negative"

        raw_transformed_dataset_filename = self.get_raw_transformed_dataset_filename()
        f = open(raw_transformed_dataset_filename, "w+")
        f.write(raw_dataset.to_csv(index=False))
        f.close()

    def get_position(self):
        name = "Position"
        feature_type = FeatureTypeCategorical()
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

    def get_oral(self):
        name = "Oral"
        feature_type = FeatureTypeContinuous()
        should_standardize = True

        return Feature(name, feature_type, should_standardize)

    def get_written(self):
        name = "Written"
        feature_type = FeatureTypeContinuous()
        should_standardize = True

        return Feature(name, feature_type, should_standardize)

    def get_race(self):
        name = "Race"
        feature_type = FeatureTypeCategorical()
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

    def get_combine(self):
        name = "Combine"
        feature_type = FeatureTypeCategorical()
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

