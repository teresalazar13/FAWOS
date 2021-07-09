from typing import List

from code.taxonomizing.Taxonomy import Taxonomy


class TaxonomyAndNeighbours:

    def __init__(self, taxonomy: Taxonomy, neighbours: List[int]):
        self.taxonomy = taxonomy
        self.neighbours = neighbours


    @staticmethod
    def read_taxonomies_and_neighbours(filename: str) -> List:
        f = open(filename, "r")
        text = f.read()

        taxonomies_and_neighbours = []
        for line in text.split("\n"):
            line = line.split(",")
            taxonomy = line[0]
            if line[1:] != ['']:
                neighbours = [int(l) for l in line[1:]]
            else:
                neighbours = [] # outlier
            taxonomy_and_neighbours = TaxonomyAndNeighbours(Taxonomy(taxonomy), neighbours)
            taxonomies_and_neighbours.append(taxonomy_and_neighbours)

        f.close()

        return taxonomies_and_neighbours


    @staticmethod
    def save_taxonomies_and_neighbours(filename: str, taxonomies_and_neighbours: List) -> None:
        f = open(filename, "w+")

        for i in range(len(taxonomies_and_neighbours)):
            f.write(taxonomies_and_neighbours[i].taxonomy.value)
            f.write(",")
            f.write(",".join([str(neighbour) for neighbour in taxonomies_and_neighbours[i].neighbours]))
            if i != len(taxonomies_and_neighbours) - 1:
                f.write("\n")

        f.close()
