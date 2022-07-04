from typing import Sequence, Callable

import numpy as np
import scipy.stats

from numpy import ndarray


def calc_entropy(vector: Sequence[float]) -> float:
    return scipy.stats.entropy(vector)


def gen_matrix(gen_value: Callable[[], float], rows: int, cols: int) -> ndarray:
    return np.array([[gen_value() for col in range(cols)] for row in range(rows)])
