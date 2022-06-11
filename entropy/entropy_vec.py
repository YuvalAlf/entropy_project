from __future__ import annotations

from itertools import chain
from typing import Callable, Iterable, Set, List

import matplotlib.pyplot as plt

from utils.core_utils import fst, snd
from utils.functional_utils import map_list, windowed_to_start
from utils.math_utils import calc_entropy


class EntropyVec(List[float]):
    @staticmethod
    def gen_rand(length: int, generator: Callable[[], float]) -> EntropyVec:
        return EntropyVec([abs(generator()) for _ in range(length)]).normalize()

    def normalize(self) -> EntropyVec:
        sum_value = sum(self)
        return EntropyVec([item / sum_value for item in self])

    def average_with(self, other_vec: EntropyVec) -> EntropyVec:
        return EntropyVec([(item1 + item2) / 2 for item1, item2 in zip(self, other_vec)])

    def top_coordinates(self, top_n: int) -> Iterable[int]:
        return map_list(fst, sorted(enumerate(self), key=snd, reverse=True))[:top_n]

    def entropy(self) -> float:
        return calc_entropy(self)

    def prepare_alg(self, other_vec: EntropyVec, top_n: int) -> (List[float], Set[int], float, float):
        other_top_coordinates = other_vec.top_coordinates(top_n)
        transmitted_coordinates = set(chain(self.top_coordinates(top_n), other_top_coordinates))
        untransmitted_coordinates = set(range(len(self))) - transmitted_coordinates
        other_known_vector = [other_vec[index] if index in transmitted_coordinates else 0 for index in range(len(self))]
        other_min_value = min(map(other_vec.__getitem__, other_top_coordinates), default=1)
        total_remaining = 1 - sum(other_known_vector)
        return other_known_vector, untransmitted_coordinates, other_min_value, total_remaining

    def upper_bound(self, other_vec: EntropyVec, top_n: int) -> float:
        other_known_vector, unknown_coordinates, _, total_remaining = self.prepare_alg(other_vec, top_n)
        values_and_locations: List[(int, float)] = sorted(((coord, self[coord]) for coord in unknown_coordinates), key=snd)
        values_and_locations.append((-1, float('inf')))
        finish_loop = False

        for ((*rest_min_values, (min_index, min_value)), (_, second_min_value)) in windowed_to_start(values_and_locations):
            amount_to_add = second_min_value - min_value
            sum_amount_to_add = amount_to_add * (len(rest_min_values) + 1)
            if sum_amount_to_add > total_remaining:
                amount_to_add = total_remaining / (len(rest_min_values) + 1)
                finish_loop = True
            total_remaining -= sum_amount_to_add

            for index, value in chain(rest_min_values, [(min_index, min_value)]):
                other_known_vector[index] += amount_to_add
            if finish_loop:
                break
        return EntropyVec(other_known_vector).average_with(self).entropy()

    def lower_bound(self, other_vec: EntropyVec, top_n: int) -> float:
        other_known_vector, unknown_coordinates, other_min_value, total_remaining = self.prepare_alg(other_vec, top_n)
        for coordinate in sorted(unknown_coordinates, key=self.__getitem__, reverse=True):
            if total_remaining <= other_min_value:
                other_known_vector[coordinate] = total_remaining
                break
            else:
                other_known_vector[coordinate] = other_min_value
                total_remaining -= other_min_value
        return EntropyVec(other_known_vector).average_with(self).entropy()

    def show_histogram(self, path: str) -> None:
        plt.plot(range(len(self)), sorted(self, reverse=True), color='red')
        plt.title('Probability Values Histogram')
        plt.ylabel('Probability Value')
        plt.xlabel('Coordinates (Sorted)')
        plt.ylim(0, None)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close('all')
