import unittest

import pandas as pd

from code.models import FeatureTypeContinuous


class TestFeatureTypeCategorical(unittest.TestCase):

    def test_encode(self):
        feature_type_continuous = FeatureTypeContinuous()

        feature_values_raw_train = pd.Series([1.2, 1.3])
        feature_values_raw_test = pd.Series([1])

        feature_values_train, feature_values_test = \
            feature_type_continuous.encode(feature_values_raw_train, feature_values_raw_test)

        pd.testing.assert_series_equal(feature_values_raw_train, pd.Series([1.2, 1.3]))
        pd.testing.assert_series_equal(feature_values_raw_test, pd.Series([1]))
        self.assertListEqual(list(feature_values_train), [1.2, 1.3])
        self.assertListEqual(list(feature_values_test), [1])
        self.assertTrue(isinstance(feature_values_train, pd.Series))
        self.assertTrue(isinstance(feature_values_test, pd.Series))


if __name__ == '__main__':
    unittest.main()