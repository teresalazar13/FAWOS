import argparse
import sys
import pandas as pd

from models.Adult import Adult
from models.Credit import Credit
from models.Ricci import Ricci
from models.dataset import Dataset


def get_dataset(dataset_name: str,
                test_size: float,
                oversampling_factor: float) -> Dataset:
    if dataset_name == "adult":
        return Adult(test_size, oversampling_factor)
    elif dataset_name == "credit":
        return Credit(test_size, oversampling_factor)
    elif dataset_name == "ricci":
        return Ricci(test_size, oversampling_factor)


def get_results(filename):
    return pd.read_csv(filename, sep=',', engine='python')


def make_tables(dataset_name: str,
                test_size: float,
                oversampling_factor: float,
                n_runs: int):

    dataset = get_dataset(dataset_name, test_size, oversampling_factor)
    results_train_list = []
    results_oversampled_list = []

    for i in range(n_runs):
        filename_train = dataset.get_train_results_filename()
        results_train = get_results(filename_train)
        results_train_list.append(results_train)

        filename_oversampled = dataset.get_train_results_filename()
        results_oversampled = get_results(filename_oversampled)
        results_oversampled_list.append(results_oversampled)

        dataset.increase_index_and_seed()

    results_train_avg = pd.concat(results_train_list).groupby(level=0).mean()
    results_oversampled_avg = pd.concat(results_oversampled_list).groupby(level=0).mean()
    print(results_train_avg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=["credit", "adult", "ricci"], required=True, help='dataset name')
    parser.add_argument('--test_size', choices=["0.2", "0.3"], required=True, help='test size')
    parser.add_argument('--oversampling_factor', required=True, help='oversampling factor')
    parser.add_argument('--n_runs', choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], required=True,
                        help='n_runs')
    args = parser.parse_args(sys.argv[1:])
    make_tables(args.dataset, args.test_size, args.oversampling_factor, int(args.n_runs))