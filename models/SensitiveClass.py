from typing import List


class SensitiveClass:

    def __init__(self,
                 name: str,
                 privileged_classes: List,
                 unprivileged_classes: List):

        self.name = name
        self.privileged_classes = privileged_classes
        self.unprivileged_classes = unprivileged_classes
