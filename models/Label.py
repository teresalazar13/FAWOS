from typing import Dict

from taxonomizing.Taxonomy import Taxonomy


class Label:

    def __init__(self,
                 target_class_value: str,
                 sensitive_class_values: Dict,
                 taxonomy: Taxonomy):
        self.target_class_value = target_class_value
        self.sensitive_class_values = sensitive_class_values
        self.taxonomy = taxonomy
