from __future__ import annotations

from itertools import chain
from typing import Callable, Iterable, Set, List

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

    def top_coordinates(self, top_n: int) -> Iterable[int]:
        return map_list(fst, sorted(enumerate(self), key=snd, reverse=True))[:top_n]

    def entropy(self) -> float:
        return calc_entropy(self)

    def prepare_alg(self, other_vec: EntropyVec, top_n: int) -> (List[float], Set[int], float):
        other_top_coordinates = other_vec.top_coordinates(top_n)
        transmitted_coordinates = set(chain(self.top_coordinates(top_n), other_top_coordinates))
        untransmitted_coordinates = set(range(len(self))) - transmitted_coordinates
        other_known_vector = [0.0] * len(self)
        for coordinate in transmitted_coordinates:
            other_known_vector[coordinate] = other_vec[coordinate]
        other_min_value = min(map(other_vec.__getitem__, other_top_coordinates), default=1)
        return other_known_vector, untransmitted_coordinates, other_min_value

    def upper_bound(self, other_vec: EntropyVec, top_n: int) -> float:
        average_known_vector, unknown_coordinates, _ = self.prepare_alg(other_vec, top_n)

        return 1

    def lower_bound(self, other_vec: EntropyVec, top_n: int) -> float:
        other_known_vector, unknown_coordinates, other_min_value = self.prepare_alg(other_vec, top_n)
        total_remaining = 1 - sum(other_known_vector)
        for coordinate in sorted(unknown_coordinates, key=self.__getitem__, reverse=True):
            if total_remaining <= other_min_value:
                other_known_vector[coordinate] = total_remaining
                break
            else:
                other_known_vector[coordinate] = other_min_value
                total_remaining -= other_min_value
        return EntropyVec(other_known_vector).average_with(self).entropy()
