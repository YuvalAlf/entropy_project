import math
from typing import Sequence


def entropy(probability: float):
    if probability <= 0:
        return 0
    return -probability * math.log(probability)


def calc_entropy(vector: Sequence[float]) -> float:
    return sum(map(entropy, vector))

