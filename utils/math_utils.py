import math
from itertools import chain
from typing import Sequence, Set, List

from utils.core_utils import snd
from utils.functional_utils import windowed_to_start


def entropy(probability: float):
    if probability <= 0:
        return 0
    return -probability * math.log(probability)


def calc_entropy(vector: Sequence[float]) -> float:
    return sum(map(entropy, vector))


def equalize_vector(current_vector: List[float], other_known_vector: List[float], unknown_coordinates: Set[int],
                    total_remaining: float) -> None:
    values_and_locations: List[(int, float)] = sorted(((coord, current_vector[coord]) for coord in unknown_coordinates),
                                                      key=snd)
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
