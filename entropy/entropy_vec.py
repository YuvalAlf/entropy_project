from __future__ import annotations

from itertools import chain
from typing import Callable, Set, List

import matplotlib.pyplot as plt
from more_itertools import pairwise

from utils.core_utils import fst, snd
from utils.functional_utils import map_list
from utils.itertools_utils import enumerate1
from utils.math_utils import calc_entropy, list_average, calc_entropy_on_average
from utils.plotting_utils import gen_plot


class EntropyVec(List[float]):
    @staticmethod
    def gen_rand(length: int, generator: Callable[[], float]) -> EntropyVec:
        return EntropyVec([abs(generator()) for _ in range(length)]).normalize()

    def normalize(self) -> EntropyVec:
        sum_value = sum(self)
        return EntropyVec([item / sum_value for item in self])

    def average_with(self, other_vec: EntropyVec) -> EntropyVec:
        return EntropyVec(list_average(self, other_vec))

    def top_coordinates(self, top_n: int) -> List[int]:
        return map_list(fst, sorted(enumerate(self), key=snd, reverse=True))[:top_n]

    def coordinates_bigger_than(self, bigger_than_value: float) -> List[int]:
        return [index for index, value in enumerate(self) if value > bigger_than_value]

    def entropy(self) -> float:
        return calc_entropy(self)

    def prepare_alg(self, other_vec: EntropyVec, top_n: int) -> (List[float], Set[int], float, float):
        other_top_coordinates = set(other_vec.top_coordinates(top_n))
        self_top_coordinates = set(self.top_coordinates(top_n))
        transmitted_coordinates = set(chain(self_top_coordinates, other_top_coordinates))
        untransmitted_coordinates = set(range(len(self))) - transmitted_coordinates
        other_known_vector = [other_vec[index] if index in transmitted_coordinates else 0 for index in range(len(self))]
        other_min_value = min(map(other_vec.__getitem__, other_top_coordinates), default=1)
        total_remaining = 1 - sum(other_known_vector)
        communication = 2 * len(transmitted_coordinates) - len(self_top_coordinates.intersection(other_top_coordinates))
        return other_known_vector, untransmitted_coordinates, other_min_value, total_remaining, communication

    def send_bigger(self, other_vec: EntropyVec, min_value: int):
        other_top_coordinates = set(other_vec.coordinates_bigger_than(min_value))
        self_top_coordinates = set(self.coordinates_bigger_than(min_value))
        transmitted_coordinates = set(chain(self_top_coordinates, other_top_coordinates))
        untransmitted_coordinates = set(range(len(self))) - transmitted_coordinates

        known_entropy = sum(calc_entropy_on_average(self[coord], other_vec[coord]) for coord in transmitted_coordinates)

        x_sum = sum(self[coord] for coord in untransmitted_coordinates)
        y_sum = sum(other_vec[coord] for coord in untransmitted_coordinates)
        x_squared_sum = sum(self[coord]**2 for coord in untransmitted_coordinates)
        y_squared_sum = sum(other_vec[coord]**2 for coord in untransmitted_coordinates)

        return known_entropy, x_sum, y_sum, x_squared_sum, y_squared_sum, untransmitted_coordinates

    def upper_bound(self, other_vec: EntropyVec, top_n: int) -> (float, int):
        other_known_vector, unknown_coordinates, _, total_remaining, communication = self.prepare_alg(other_vec, top_n)
        average_vector = self.average_with(other_known_vector)
        if len(unknown_coordinates) == 0:
            return average_vector.entropy(), communication
        average_vector_sorted_values = sorted(map(average_vector.__getitem__, unknown_coordinates)) + [float('inf')]
        addition_remaining_to_average_vec = total_remaining / 2

        for width, (prev_value, next_value) in enumerate1(pairwise(average_vector_sorted_values)):
            height = next_value - prev_value
            addition_value = width * height
            if addition_value < addition_remaining_to_average_vec:
                addition_remaining_to_average_vec -= addition_value
            else:
                height = addition_remaining_to_average_vec / width
                threshold_value = prev_value + height
                for coord in unknown_coordinates:
                    if average_vector[coord] < threshold_value:
                        average_vector[coord] = threshold_value
                return average_vector.entropy(), communication

        raise ValueError("Some unexpected error occurred. shouldn't reach here")

    def lower_bound(self, other_vec: EntropyVec, top_n: int) -> (float, int):
        other_known_vector, unknown_coordinates, other_min_value, total_remaining, communication = self.prepare_alg(other_vec, top_n)
        for coordinate in sorted(unknown_coordinates, key=self.__getitem__, reverse=True):
            if total_remaining <= other_min_value:
                other_known_vector[coordinate] = total_remaining
                break
            else:
                other_known_vector[coordinate] = other_min_value
                total_remaining -= other_min_value
        return EntropyVec(other_known_vector).average_with(self).entropy(), communication

    def show_histogram(self, save_path: str) -> None:
        with gen_plot(save_path, x_label='Coordinates (Sorted)', y_label='Probability Value',
                      title='Probability Values Histogram'):
            plt.plot(range(len(self)), sorted(self, reverse=True), color='red')
            plt.ylim(0, None)
