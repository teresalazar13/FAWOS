from typing import List, Dict

from oversampling.DatapointsToOversample import DatapointsToOversample


class DatapointsFromClassToOversample:

    def __init__(self, n_times_to_oversample: int, datapoints_to_oversample_list: List[DatapointsToOversample], classes: Dict) -> None:
        self.n_times_to_oversample = n_times_to_oversample
        self.datapoints_to_oversample_list = datapoints_to_oversample_list
        self.classes = classes
