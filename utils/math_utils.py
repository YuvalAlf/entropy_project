import math
from typing import Sequence, Callable, List, Iterable

import numpy as np
import scipy.stats

from numpy import ndarray


def calc_entropy(vector: Sequence[float]) -> float:
    return scipy.stats.entropy(vector)


def calc_entropy_on_average(x: float, y: float) -> float:
    average = (x + y) / 2
    if average == 0:
        return 0
    return -average * math.log(average)


def calc_entropy_on_average_multiple(*values: float) -> float:
    average = sum(values) / len(values)
    if average == 0:
        return 0
    return -average * math.log(average)


def gen_matrix(gen_value: Callable[[], float], rows: int, cols: int) -> ndarray:
    return np.array([[gen_value() for col in range(cols)] for row in range(rows)])


def inner_product(vec1: Iterable[float], vec2: Iterable[float]) -> float:
    return sum(a * b for a, b in zip(vec1, vec2))


def list_average(lst1: List[float], lst2: List[float]) -> List[float]:
    return [(item1 + item2) / 2 for item1, item2 in zip(lst1, lst2)]
