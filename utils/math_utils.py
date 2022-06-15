from typing import Sequence

import scipy.stats


# def entropy_value(probability: float):
#     if probability <= 0:
#         return 0
#     return -probability * math.log(probability)


def calc_entropy(vector: Sequence[float]) -> float:
    return scipy.stats.entropy(vector)
