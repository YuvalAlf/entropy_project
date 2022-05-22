import math

from numpy import ndarray


def entropy(probability: float):
    if probability < 0:
        return 0
    return -probability * math.log(probability)


def calc_entropy(vector: ndarray) -> float:
    return sum(map(entropy, vector))
