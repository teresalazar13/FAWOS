from models.FeatureType import FeatureType


class Feature:

    def __init__(self,
                 name: str,
                 feature_type: FeatureType,
                 should_standardize: bool):

        self.name = name
        self.feature_type = feature_type
        self.should_standardize = should_standardize
