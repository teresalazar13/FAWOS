from typing import List

from oversampling.DatapointAndNeighbours import DatapointAndNeighbours
from taxonomizing import Taxonomy


class DatapointsToOversample:

    def __init__(self, taxonomy: Taxonomy, datapoints_and_neighbours: List[DatapointAndNeighbours]) -> None:
        self.taxonomy = taxonomy
        self.datapoints_and_neighbours = datapoints_and_neighbours
