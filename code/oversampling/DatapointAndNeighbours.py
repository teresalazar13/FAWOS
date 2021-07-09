import numpy as np
from typing import List


class DatapointAndNeighbours:

    def __init__(self, datapoint: np.array, neighbours: List[np.array]) -> None:
        self.datapoint = datapoint
        self.neighbours = neighbours
