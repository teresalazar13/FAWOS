import pandas as pd

from models.Feature import Feature
from models.FeatureTypeCategorical import FeatureTypeCategorical
from models.FeatureTypeContinuous import FeatureTypeContinuous
from models.FeatureTypeOrdinal import FeatureTypeOrdinal
from models.SensitiveClass import SensitiveClass
from models.TargetClass import TargetClass
from models.dataset import Dataset


class Credit(Dataset):
    # https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

    def __init__(self):
        name = "credit"
        target_class = TargetClass("credit", "Positive", "Negative")
        # sensitive_class_gender = SensitiveClass("personal_status", ["male"], ["female"]) # there's no A95??? -> no single women
        sensitive_class_age = SensitiveClass("age", ["adult"], ["young"])
        sensitive_classes = [sensitive_class_age]  # TODO add gender
        features = [
            self.get_feature_status(),
            self.get_feature_month(),
            self.get_feature_credit_history(),
            self.get_feature_purpose(),
            self.get_feature_credit_amount(),
            self.get_feature_savings(),
            self.get_feature_employment(),
            self.get_feature_investment_as_income_percentage(),
            self.get_feature_personal_status(),
            self.get_feature_other_debtors(),
            self.get_feature_residence_since(),
            self.get_feature_property(),
            self.get_feature_age(),
            self.get_feature_installment_plans(),
            self.get_feature_housing(),
            self.get_feature_number_of_credits(),
            self.get_feature_skill_level(),
            self.get_feature_people_liable_for(),
            self.get_feature_telephone(),
            self.get_feature_foreign_worker(),
            self.get_feature_credit()
        ]

        super().__init__(name, target_class, sensitive_classes, features)

    def create_raw_transformed_dataset(self):
        raw_dataset = self.get_raw_dataset()

        old = raw_dataset['age'] >= 25  # http://ieeexplore.ieee.org/document/4909197/
        raw_dataset.loc[old, 'age'] = "adult"
        young = raw_dataset['age'] != "adult"
        raw_dataset.loc[young, 'age'] = "young"

        positive = raw_dataset['credit'] == 1
        raw_dataset.loc[positive, 'credit'] = "Positive"
        negative = raw_dataset['credit'] == 2
        raw_dataset.loc[negative, 'credit'] = "Negative"

        raw_transformed_dataset_filename = self.get_raw_transformed_dataset_filename()
        f = open(raw_transformed_dataset_filename, "w+")
        f.write(raw_dataset.to_csv(index=False))
        f.close()

    @staticmethod
    def get_feature_status():
        name = "status"
        order = ["A11", "A14", "A12", "A13"]
        feature_type = FeatureTypeOrdinal(order)
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_month():
        name = "month"
        feature_type = FeatureTypeContinuous()
        should_standardize = True

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_credit_history():
        name = "credit_history"
        feature_type = FeatureTypeCategorical()
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_purpose():
        name = "purpose"
        feature_type = FeatureTypeCategorical()
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_credit_amount():
        name = "credit_amount"
        feature_type = FeatureTypeContinuous()
        should_standardize = True

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_savings():
        name = "savings"
        order = ["A65", "A61", "A62", "A63", "A64"]
        feature_type = FeatureTypeOrdinal(order)
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_employment():
        name = "employment"
        order = ["A71", "A72", "A73", "A74", "A75"]
        feature_type = FeatureTypeOrdinal(order)
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_investment_as_income_percentage():
        name = "investment_as_income_percentage"
        feature_type = FeatureTypeContinuous()
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_personal_status():
        name = "personal_status"
        feature_type = FeatureTypeCategorical()
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_other_debtors():
        name = "other_debtors"
        feature_type = FeatureTypeCategorical()
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_residence_since():
        name = "residence_since"
        feature_type = FeatureTypeContinuous()
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_property():
        name = "property"
        feature_type = FeatureTypeCategorical()
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_age():
        name = "age"
        feature_type = FeatureTypeCategorical()
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_installment_plans():
        name = "installment_plans"
        feature_type = FeatureTypeCategorical()
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_housing():
        name = "housing"
        feature_type = FeatureTypeCategorical()
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_number_of_credits():
        name = "number_of_credits"
        feature_type = FeatureTypeContinuous()
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_skill_level():
        name = "skill_level"
        order = ["A171", "A172", "A173", "A174"]
        feature_type = FeatureTypeOrdinal(order)
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_people_liable_for():
        name = "people_liable_for"
        feature_type = FeatureTypeContinuous()
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_telephone():
        name = "telephone"
        feature_type = FeatureTypeCategorical()
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_foreign_worker():
        name = "foreign_worker"
        feature_type = FeatureTypeCategorical()
        should_standardize = False

        return Feature(name, feature_type, should_standardize)

    @staticmethod
    def get_feature_credit():
        name = "credit"
        feature_type = FeatureTypeCategorical()
        should_standardize = False

        return Feature(name, feature_type, should_standardize)
