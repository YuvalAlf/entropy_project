from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy import ndarray

from utils.math_utils import calc_entropy


@dataclass
class EntropyVec:
    vector: ndarray

    def entropy(self) -> float:
        return calc_entropy(self.vector)

    def upper_bound(self, other_vec: EntropyVec, top_n: int) -> float:
        return 1

    def lower_bound(self, other_vec: EntropyVec, top_n: int) -> float:
        return 2

    @staticmethod
    def gen_rand(length: int, generator: Callable[[], float]) -> EntropyVec:
        return EntropyVec(np.array([generator() for _ in range(length)]))
