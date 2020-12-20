import json
from operator import attrgetter
from typing import List

from models.Label import Label
from taxonomizing.Taxonomy import Taxonomy


class Distribution:

    def __init__(self,
                 label: Label,
                 count: int,
                 percentage: float):
        self.label = label
        self.count = count
        self.percentage = percentage

    def __eq__(self, other):
        return self.label == other.label \
               and self.count == other.count \
               and self.percentage == other.percentage

    def __hash__(self):
        return hash(('label', self.label, 'count', self.count, 'percentage', self.percentage))


    @staticmethod
    def read_distributions(filename: str):
        f = open(filename, "r")
        text = f.read()

        distributions = []
        for line in text.split("\n"):
            if line:
                sensitive_class_values = "{" + line.split("{")[1]
                sensitive_class_values = sensitive_class_values.replace(sensitive_class_values.split("}")[1], "")
                sensitive_class_values = json.loads(sensitive_class_values)
                line = line.split(",")
                target_class_value = line[0]
                taxonomy = Taxonomy(line[-3])
                count = int(line[-2])
                percentage = float(line[-1])
                label = Label(target_class_value, sensitive_class_values, taxonomy)
                distributions.append(Distribution(label, count, percentage))

        f.close()

        return distributions


    @staticmethod
    def save_distributions(labels: List[Label], stats_filename: str):
        labels_strings = []
        labels = sorted(labels, key=lambda x: (" ".join(x.sensitive_class_values), x.target_class_value, x.taxonomy.value))

        for label in labels:
            labels_string = label.target_class_value
            labels_string += ","
            labels_string += json.dumps(label.sensitive_class_values)
            labels_string += ","
            labels_string += label.taxonomy.value
            labels_string += ","
            labels_strings.append(labels_string)

        f = open(stats_filename, "w+")
        for unique_label_string in set(labels_strings):
            f.write(unique_label_string)
            count = labels_strings.count(unique_label_string)
            f.write(str(count))
            f.write(",")
            f.write(str(round(count / len(labels_strings), 2)))
            f.write("\n")
        f.close()
