from dataclasses import dataclass
from functools import cached_property
from random import Random

import numpy as np
from numpy import ndarray

from utils.math_utils import gen_matrix


@dataclass
class InnerProductSketch:
    sketch_size: int
    vector_size: int
    prng: Random

    @cached_property
    def projection_matrix(self) -> ndarray:
        rows = self.sketch_size
        cols = self.vector_size
        return gen_matrix(lambda: self.prng.uniform(-1, 1), rows, cols)

    def calc_inner_product(self, vec1: ndarray, vec2: ndarray) -> float:
        projection1 = self.projection_matrix.dot(vec1)
        projection2 = self.projection_matrix.dot(vec2)
        return np.inner(projection1, projection2)
