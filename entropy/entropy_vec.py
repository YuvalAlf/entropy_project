from __future__ import annotations

from itertools import chain
from typing import Callable, Iterable, Set, List

import numpy as np
from numpy import ndarray

from utils.core_utils import fst, snd
from utils.functional_utils import map_list
from utils.math_utils import calc_entropy


class EntropyVec(List[float]):
    @staticmethod
    def gen_rand(length: int, generator: Callable[[], float]) -> EntropyVec:
        return EntropyVec([generator() for _ in range(length)]).normalize()

    def normalize(self) -> EntropyVec:
        sum_value = sum(self)
        return EntropyVec([item / sum_value for item in self])

    def average_with(self, other_vec: EntropyVec) -> EntropyVec:
        return EntropyVec([(item1 + item2) / 2 for item1, item2 in zip(self, other_vec)])

    def top_locations(self, top_n: int) -> Iterable[int]:
        return map_list(fst, sorted(enumerate(self), key=snd, reverse=True))[:top_n]

    def entropy(self) -> float:
        return calc_entropy(self)

    def prepare_alg(self, other_vec: EntropyVec, top_n: int) -> (List[float], Set[int], float, float):
        other_top_locations = other_vec.top_locations(top_n)
        other_sum_remaining = 1 - sum(map(other_vec.__getitem__, other_top_locations))
        known_locations = set(chain(self.top_locations(top_n), other_top_locations))
        unknown_locations = set(range(len(self))) - known_locations
        average_known_vector = [0.0] * len(self)
        for location in known_locations:
            average_known_vector[location] = (self[location] + other_vec[location]) / 2
        other_min_value = min(map(other_vec.__getitem__, other_top_locations), default=1)
        return average_known_vector, unknown_locations, other_sum_remaining, other_min_value

    def upper_bound(self, other_vec: EntropyVec, top_n: int) -> float:
        average_known_vector, unknown_locations, other_sum_remaining, _ = self.prepare_alg(other_vec, top_n)

        return 1

    def lower_bound(self, other_vec: EntropyVec, top_n: int) -> float:
        average_known_vector, unknown_locations, other_sum_remaining, other_min_value = self.prepare_alg(other_vec, top_n)
        total_remaining = other_sum_remaining
        for location in sorted(unknown_locations, key=self.__getitem__, reverse=True):
            if total_remaining <= other_min_value:
                average_known_vector[location] = total_remaining
                break
            else:
                average_known_vector[location] = other_min_value
                total_remaining -= other_min_value
        return EntropyVec(average_known_vector).entropy()
