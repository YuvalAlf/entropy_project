from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass
from math import pi
from random import Random
from statistics import mean
from typing import List

from sketches.projection_sketch import ProjectionSketch
from utils.functional_utils import map_list
from utils.math_utils import list_average


@dataclass
class CliffordEntropySketch(ProjectionSketch):
    @staticmethod
    def stable_distribution(prng: Random) -> float:
        u1, u2 = prng.uniform(0, 1), prng.uniform(0, 1)
        w1 = pi * (u1 - 0.5)
        w2 = -math.log(u2)
        return math.tan(w1) * (pi / 2 - w1) + math.log(w2 * math.cos(w1) / (pi / 2 - w1))

    def gen_projection_vector(self, prng: Random) -> List[float]:
        return [CliffordEntropySketch.stable_distribution(prng) for _ in range(self.vector_size)]

    def sketch_calculation(self, sketch1: List[float], sketch2: List[float]) -> float:
        average_sketch = list_average(sketch1, sketch2)
        return -math.log(mean(map_list(math.exp, average_sketch)))
