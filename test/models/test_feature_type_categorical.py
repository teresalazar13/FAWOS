import unittest
import pandas as pd

from models.FeatureTypeCategorical import FeatureTypeCategorical


class TestFeatureTypeCategorical(unittest.TestCase):

    def test_encode(self):
        feature_type_categorical = FeatureTypeCategorical()

        feature_values_raw_train = pd.Series(["hello", "goodbye", "hi"])
        feature_values_raw_test = pd.Series(["hello", "hi"])

        feature_values_train, feature_values_test = \
            feature_type_categorical.encode(feature_values_raw_train, feature_values_raw_test)

        pd.testing.assert_series_equal(feature_values_raw_train, pd.Series(["hello", "goodbye", "hi"]))
        pd.testing.assert_series_equal(feature_values_raw_test, pd.Series(["hello", "hi"]))
        self.assertListEqual(list(feature_values_train), [1, 0, 2])
        self.assertListEqual(list(feature_values_test), [1, 2])
        self.assertTrue(isinstance(feature_values_train, pd.Series))
        self.assertTrue(isinstance(feature_values_test, pd.Series))


if __name__ == '__main__':
    unittest.main()