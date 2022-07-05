from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from math import pi
from random import Random
from statistics import mean
from typing import List, Iterable, Tuple

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
        # TODO: generate vector after vector
        rows = self.sketch_size
        cols = self.vector_size
        return gen_matrix(self.stable_distribution, rows, cols)

    def project(self, prob_vector: List[float]) -> ndarray:
        return self.projection_matrix.dot(np.asarray(prob_vector))

    def apply(self, prob_vector: List[float]) -> float:
        return -math.log(mean(map_list(math.exp, self.project(prob_vector))))

    def sketch_approximations(self, prob_vector: List[float]) -> Iterable[Tuple[int, float]]:
        exponent_values = map_list(math.exp, self.project(prob_vector))
        cumsum_exponent_values = cumsum(exponent_values)
        for sketch_size, cumsum_value in enumerate1(cumsum_exponent_values):
            sketch_value = 0 if cumsum_value <= 0 else -math.log(cumsum_value / sketch_size)
            yield sketch_size, sketch_value
