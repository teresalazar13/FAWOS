import unittest

import pandas as pd
from pandas._testing import assert_frame_equal

from code.preprocessing import preprocessor


class TestPreprocessor(unittest.TestCase):

    def test_standardize(self):
        train_dataset = pd.DataFrame({"index": [1, 2, 3, 4, 5], "data": [1, 2, 3, 4, 5], "oi": [2, 2, 2, 2, 2]})
        test_dataset = pd.DataFrame({"index": [1, 2, 3, 4], "data": [-1, 3, 5, 6], "oi": [3, 3, 3, 3]})

        features_to_standardize = ["data"]
        train_dataset[features_to_standardize], test_dataset[features_to_standardize] \
            = preprocessor.standardize(train_dataset[features_to_standardize], test_dataset[features_to_standardize])

        train_dataset_expected = pd.DataFrame({"index": [1, 2, 3, 4, 5], "data":[-1.414214, -0.707107, 0.000000, 0.707107, 1.414214], "oi": [2, 2, 2, 2, 2]})
        test_dataset_expected = pd.DataFrame({"index": [1, 2, 3, 4], "data":[-2.828427, 0.000000, 1.414214, 2.121320], "oi": [3, 3, 3, 3]})
        assert_frame_equal(train_dataset, train_dataset_expected)
        assert_frame_equal(test_dataset, test_dataset_expected)


if __name__ == '__main__':
    unittest.main()