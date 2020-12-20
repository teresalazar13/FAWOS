from models.Feature import Feature
from models.FeatureTypeCategorical import FeatureTypeCategorical
from models.FeatureTypeContinuous import FeatureTypeContinuous
from models.FeatureTypeOrdinal import FeatureTypeOrdinal
from models.SensitiveClass import SensitiveClass
from models.TargetClass import TargetClass
from models.dataset import Dataset


class Adult(Dataset):

    def __init__(self, test_size, oversampling_factor):
        name = "adult"
        target_class = TargetClass("income", ">50K", "<=50K")
        sensitive_class_gender = SensitiveClass("gender", ["Male"], ["Female"])
        sensitive_classes = [sensitive_class_gender]  # TODO add race
        features = [
            self.get_feature_age(),
            self.get_feature_workclass(),
            self.get_feature_fnlwgt(),
            self.get_feature_education(),
            self.get_feature_educational_num(),
            self.get_feature_marital_status(),
            self.get_feature_occupation(),
            self.get_feature_relationship(),
            self.get_feature_race(),
            self.get_feature_gender(),
            self.get_feature_capital_gain(),
            self.get_feature_capital_loss(),
            self.get_feature_hours_per_week(),
            self.get_feature_native_country(),
            self.get_feature_income()
        ]

        super().__init__(name, target_class, sensitive_classes, features, test_size, oversampling_factor)

    def create_raw_transformed_dataset(self):
        super().create_raw_transformed_dataset()

    @staticmethod
    def get_feature_age():
        name = "age"
        feature_type = FeatureTypeContinuous()
        should_standardize = True

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_workclass():
        name = "workclass"
        feature_type = FeatureTypeCategorical()
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_fnlwgt():
        name = "fnlwgt"
        feature_type = FeatureTypeContinuous()
        should_standardize = True

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_education():
        name = "education"
        order = ["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th", "HS-grad", "Assoc-voc", "Assoc-acdm", "Prof-school", "Some-college", "Bachelors", "Masters", "Doctorate"] # TODO add order
        feature_type = FeatureTypeOrdinal(order)
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_educational_num():
        name = "educational-num"
        feature_type = FeatureTypeContinuous()
        should_standardize = True

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_marital_status():
        name = "marital-status"
        feature_type = FeatureTypeCategorical()
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_occupation():
        name = "occupation"
        feature_type = FeatureTypeCategorical()
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_relationship():
        name = "relationship"
        feature_type = FeatureTypeCategorical()
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_race():
        name = "race"
        feature_type = FeatureTypeCategorical()
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_gender():
        name = "gender"
        feature_type = FeatureTypeCategorical()
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_capital_gain():
        name = "capital-gain"
        feature_type = FeatureTypeContinuous()
        should_standardize = True

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_capital_loss():
        name = "capital-loss"
        feature_type = FeatureTypeContinuous()
        should_standardize = True

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_hours_per_week():
        name = "hours-per-week"
        feature_type = FeatureTypeContinuous()
        should_standardize = True

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_native_country():
        name = "native-country"
        feature_type = FeatureTypeCategorical()
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_income():
        name = "income"
        feature_type = FeatureTypeCategorical()
        should_standardize = False

        return Feature(name, feature_type, should_standardize)
