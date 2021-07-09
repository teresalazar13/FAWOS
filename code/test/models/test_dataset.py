import unittest
import pandas as pd

from code.models import Dataset


class TestDataset(unittest.TestCase):

    def test_map_feature_value_to_raw_string_value(self):
        df_raw = pd.DataFrame({"values": ["Male", "Female", "Male", "Female"]})
        df = pd.DataFrame({"values": [0, 1, 0, 1]})

        value_male = Dataset.map_feature_value_to_raw_string_value(df_raw, df, "values", "Male")
        self.assertEqual(0, value_male)

        value_female = Dataset.map_feature_value_to_raw_string_value(df_raw, df, "values", "Female")
        self.assertEqual(1, value_female)


if __name__ == '__main__':
    unittest.main()