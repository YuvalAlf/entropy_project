from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from math import pi
from random import Random
from statistics import mean
from typing import List

import numpy as np
from numpy import ndarray, cumsum

from utils.functional_utils import map_list
from utils.itertools_utils import enumerate1
from utils.math_utils import gen_matrix


@dataclass
class EntropySketch:
    sketch_size: int
    vector_size: int
    prng: Random

    def stable_distribution(self) -> float:
        u1, u2 = self.prng.uniform(0, 1), self.prng.uniform(0, 1)
        w1 = pi * (u1 - 0.5)
        w2 = -math.log(u2)
        return math.tan(w1) * (pi / 2 - w1) + math.log(w2 * math.cos(w1) / (pi / 2 - w1))

    @cached_property
    def projection_matrix(self) -> ndarray:
        rows = self.sketch_size
        cols = self.vector_size
        return gen_matrix(self.stable_distribution(), rows, cols)

    def apply(self, prob_vector: List[float]) -> (float, List[float]):
        sketch = self.projection_matrix.dot(np.asarray(prob_vector))
        approximated_entropy = -math.log(mean(map_list(math.exp, sketch)))
        return approximated_entropy, sketch

    @staticmethod
    def sketch_approximations(average_sketch: List[float]):
        exponent_values = map_list(math.exp, average_sketch)
        cumsum_exponent_values = cumsum(exponent_values)
        for index, cumsum_value in enumerate1(cumsum_exponent_values):
            yield -math.log(cumsum_value / index)
