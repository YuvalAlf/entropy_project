import math
import os
from dataclasses import dataclass
from typing import Tuple, Iterable, List

import matplotlib.pyplot as plt
import numpy as np

from entropy.entropy_vec import EntropyVec
from sketches.jl_sketch import JohnsonLindenstraussSketch
from utils.combinatorics_utils import combinations_without_repetitions
from utils.functional_utils import map_list
from utils.math_utils import inner_product
from utils.os_utils import join_create_dir
from utils.paths_dir import RESULTS_DIR_PATH
from utils.plotting_utils import gen_plot


@dataclass
class EntropyPolyApproximationDeg2:
    epsilon: float

    def calc_approximation(self, x: float) -> float:
        return -5/4/self.epsilon*x*x+(13/12-math.log(self.epsilon))*x+self.epsilon/8.0

    def calc_approximation_on_vector(self, x_sum: float, y_sum: float, x_squared_sum: float, y_squared_sum: float,
                                     length: int, inner_product_value: float) -> float:
        return self.epsilon/8.0 * length + ((13/12-math.log(self.epsilon)) / 2) * (x_sum + y_sum) - (5/4/self.epsilon / 4) * (x_squared_sum + y_squared_sum + 2 * inner_product_value)

    def calc_approximation_on_vectors_multiple(self, sums: List[float], squared_sums: List[float],
                                               sketches: List[List[float]], length: int) -> float:
        inner_products_sums = sum(inner_product(sketch1, sketch2) / len(sketch1) for sketch1, sketch2 in combinations_without_repetitions(sketches))
        return self.epsilon/8.0 * length + (13/12-math.log(self.epsilon)) * sum(sums) / len(sums) - (5/4/self.epsilon / (len(sums) ** 2)) * (sum(squared_sums) + 2 * inner_products_sums)

    def calc_real_value(self, x: float) -> float:
        if x == 0:
            return 0
        return -x * math.log(x)

    def draw_approximation(self, save_dir: str) -> None:
        xs = list(np.linspace(0, self.epsilon, 1000))
        real_ys = map_list(self.calc_real_value, xs)
        approximated_ys = map_list(self.calc_approximation, xs)
        with gen_plot(os.path.join(save_dir, f'approximation.png'),
                      width=8, height=6.5, x_label='X', y_label='Y',
                      title=f'Entropy Polynomial Approximation, Epsilon = {self.epsilon}'):
            plt.plot(xs, real_ys, alpha=0.7, label='Entropy', color='r')
            plt.plot(xs, approximated_ys, alpha=0.7, label='Poly', color='b')
            plt.legend()

    def draw(self, sketch_size: int, probability_vector1: EntropyVec, probability_vector2: EntropyVec, color: str, label: str, seed: int) -> None:
        jl_sketch = JohnsonLindenstraussSketch(sketch_size, len(probability_vector1), seed)
        xs = []
        ys = []
        known_entropy, x_sum, y_sum, x_squared_sum, y_squared_sum, x_tripled_sum, y_tripled_sum, untransmitted_coords = probability_vector1.send_bigger(probability_vector2, min_value=self.epsilon)
        inner_prod_vector1 = [probability_vector1[coord] for coord in untransmitted_coords]
        inner_prod_vector2 = [probability_vector2[coord] for coord in untransmitted_coords]
        for sketch_size, inner_product_value in jl_sketch.sketch_approximations(inner_prod_vector1, inner_prod_vector2):
            xs.append(sketch_size)
            ys.append(known_entropy + self.calc_approximation_on_vector(x_sum, y_sum, x_squared_sum, y_squared_sum, len(untransmitted_coords), inner_product_value))
        plt.plot(xs, ys, color=color, label=label)

    def sketch_approximations(self, sketch_size: int, probability_vector1: EntropyVec, probability_vector2: EntropyVec, seed: int) -> Iterable[Tuple[int, float]]:
        jl_sketch = JohnsonLindenstraussSketch(sketch_size, len(probability_vector1), seed)
        known_entropy, x_sum, y_sum, x_squared_sum, y_squared_sum, x_tripled_sum, y_tripled_sum, untransmitted_coords = probability_vector1.send_bigger(probability_vector2, min_value=self.epsilon)
        inner_prod_vector1 = [probability_vector1[coord] for coord in untransmitted_coords]
        inner_prod_vector2 = [probability_vector2[coord] for coord in untransmitted_coords]
        for sketch_size, inner_product_value in jl_sketch.sketch_approximations(inner_prod_vector1, inner_prod_vector2):
            x = sketch_size
            y = known_entropy + self.calc_approximation_on_vector(x_sum, y_sum, x_squared_sum, y_squared_sum, len(untransmitted_coords), inner_product_value)
            yield x, y

    def sketch_approximations_multiple(self, sketch_size: int, probability_vectors: List[EntropyVec], seed: int) -> Iterable[Tuple[int, float]]:
        jl_sketch = JohnsonLindenstraussSketch(sketch_size, len(probability_vectors[0]), seed)
        known_entropy, sums, sums_squared, untransmitted_coords = EntropyVec.send_bigger_multiple(probability_vectors, min_value=self.epsilon)
        rest_vectors = [[prob_vector[coord] for coord in untransmitted_coords] for prob_vector in probability_vectors]
        for sketch_size, sketches in jl_sketch.sketch_approximations_multiple(rest_vectors):
            approximated_entropy = known_entropy + self.calc_approximation_on_vectors_multiple(sums, sums_squared, sketches, len(untransmitted_coords))
            yield sketch_size, approximated_entropy

    def draw_communication(self, max_sketch_size: int, probability_vector1: EntropyVec, probability_vector2: EntropyVec,
                           color: str, label: str, seed: int) -> float:
        jl_sketch = JohnsonLindenstraussSketch(max_sketch_size, len(probability_vector1), seed)
        xs = []
        ys = []
        known_entropy, x_sum, y_sum, x_squared_sum, y_squared_sum, x_tripled_sum, y_tripled_sum, untransmitted_coords = probability_vector1.send_bigger(probability_vector2, min_value=self.epsilon)
        num_transmitted_coords = len(probability_vector1) - len(untransmitted_coords)
        vector1 = [probability_vector1[coord] for coord in untransmitted_coords]
        vector2 = [probability_vector2[coord] for coord in untransmitted_coords]
        xy_sketch = jl_sketch.sketch_approximations(vector1, vector2)
        for sketch_size, xy in xy_sketch:
            xs.append(num_transmitted_coords * 4 + sketch_size * 2 + 2)
            ys.append(known_entropy + self.calc_approximation_on_vector(x_sum, y_sum, x_squared_sum, y_squared_sum, len(untransmitted_coords), xy))
        plt.plot(xs, ys, color=color, label=label, alpha=0.8)
        return max(xs)


if __name__ == '__main__':
    EntropyPolyApproximationDeg2(0.0001).draw_approximation(join_create_dir('..', RESULTS_DIR_PATH, 'entropy_poly_approximation'))
