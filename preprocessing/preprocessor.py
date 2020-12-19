import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from models.dataset import Dataset


def apply_specific_dataset_processing(dataset: Dataset):
    dataset.create_raw_transformed_dataset()


def train_test_split_and_save(dataset: Dataset):
    raw_dataset = dataset.get_raw_transformed_dataset()
    X = raw_dataset.loc[:, raw_dataset.columns != dataset.target_class.name]
    y = raw_dataset[dataset.target_class.name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=dataset.test_size, random_state=dataset.seed)

    raw_train_dataset = pd.concat((X_train, pd.Series(y_train).rename(dataset.target_class.name)), axis=1)
    raw_test_dataset = pd.concat((X_test, pd.Series(y_test).rename(dataset.target_class.name)), axis=1)
    save_dataset(raw_train_dataset, dataset.get_raw_train_dataset_filename())
    save_dataset(raw_test_dataset, dataset.get_raw_test_dataset_filename())

    return X_train, X_test, y_train, y_test


def create_preprocessed_train_and_test_datasets(dataset: Dataset,
                                                raw_train_dataset_X,
                                                raw_test_dataset_X,
                                                raw_train_dataset_y,
                                                raw_test_dataset_y):
    features = dataset.features
    train_dataset = pd.DataFrame.copy(raw_train_dataset_X, deep=True)
    train_dataset = pd.concat((train_dataset, pd.Series(raw_train_dataset_y).rename(dataset.target_class.name)), axis=1)
    test_dataset = pd.DataFrame.copy(raw_test_dataset_X, deep=True)
    test_dataset = pd.concat((test_dataset, pd.Series(raw_test_dataset_y).rename(dataset.target_class.name)), axis=1)

    # Reset Encoding Mapping
    dataset.reset_encoding_mapping()

    for feature in features:
        feature_name = feature.name
        feature_values_raw_train = train_dataset[feature_name]
        feature_values_raw_test = test_dataset[feature_name]

        # Encode and Save Encoding Mapping
        feature_values_train, feature_values_test = \
            feature.feature_type.encode(dataset, feature_name, feature_values_raw_train, feature_values_raw_test)

        train_dataset[feature_name] = feature_values_train
        test_dataset[feature_name] = feature_values_test

    # Standardize
    features_to_standardize = [f.name for f in features if f.should_standardize]
    train_dataset[features_to_standardize], test_dataset[features_to_standardize] \
        = standardize(train_dataset[features_to_standardize], test_dataset[features_to_standardize])

    save_dataset(train_dataset, dataset.get_train_dataset_filename())
    save_dataset(test_dataset, dataset.get_test_dataset_filename())


def standardize(train_dataset: pd.DataFrame,
                test_dataset: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame]:

    scaler = StandardScaler().fit(train_dataset)
    train_dataset_scaled = scaler.transform(train_dataset)
    test_dataset_scaled = scaler.transform(test_dataset)

    train_dataset_temp = pd.DataFrame(train_dataset_scaled, columns=train_dataset.columns, index=train_dataset.index)
    test_dataset_temp = pd.DataFrame(test_dataset_scaled, columns=test_dataset.columns, index=test_dataset.index)

    return train_dataset_temp, test_dataset_temp


def save_dataset(dataset: pd.DataFrame, filename: str):
    f = open(filename, "w+")
    f.write(dataset.to_csv(index=False))
    f.close()
