from typing import Sequence, Callable, List, Iterable

import numpy as np
import scipy.stats

from numpy import ndarray


def calc_entropy(vector: Sequence[float]) -> float:
    return scipy.stats.entropy(vector)


def gen_matrix(gen_value: Callable[[], float], rows: int, cols: int) -> ndarray:
    return np.array([[gen_value() for col in range(cols)] for row in range(rows)])


def inner_product(vec1: Iterable[float], vec2: Iterable[float]) -> float:
    return sum(a * b for a, b in zip(vec1, vec2))


def list_average(lst1: List[float], lst2: List[float]) -> List[float]:
    return [(item1 + item2) / 2 for item1, item2 in zip(lst1, lst2)]
