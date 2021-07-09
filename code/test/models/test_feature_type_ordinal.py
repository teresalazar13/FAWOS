import unittest
import pandas as pd

from code.models import FeatureTypeOrdinal


class TestFeatureTypeOrdinal(unittest.TestCase):

    def test_encode(self):
        order = ["one", "two", "three", "four"]
        feature_type_categorical = FeatureTypeOrdinal(order)

        feature_values_raw_train = pd.Series(["one", "two", "three"])
        feature_values_raw_test = pd.Series(["four", "one"])

        feature_values_train, feature_values_test = \
            feature_type_categorical.encode(feature_values_raw_train, feature_values_raw_test)

        pd.testing.assert_series_equal(feature_values_raw_train, pd.Series(["one", "two", "three"]))
        pd.testing.assert_series_equal(feature_values_raw_test, pd.Series(["four", "one"]))
        self.assertListEqual(list(feature_values_train), [0, 1, 2])
        self.assertListEqual(list(feature_values_test), [3, 0])
        self.assertTrue(isinstance(feature_values_train, pd.Series))
        self.assertTrue(isinstance(feature_values_test, pd.Series))


if __name__ == '__main__':
    unittest.main()