import math
import os
from dataclasses import dataclass
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np

from entropy.entropy_vec import EntropyVec
from sketches.jl_sketch import JohnsonLindenstraussSketch
from utils.functional_utils import map_list
from utils.math_utils import inner_product
from utils.os_utils import join_create_dir
from utils.paths_dir import RESULTS_DIR_PATH
from utils.plotting_utils import gen_plot


@dataclass
class EntropyPolyApproximation2Deg3:
    epsilon: float

    def calc_approximation(self, x: float) -> float:
        return 7/6/(self.epsilon ** 2) * x * x * x - 3/self.epsilon*x*x + (107/60 - math.log(self.epsilon)) * x + self.epsilon / 15

    def calc_approximation_on_vector(self, x_sum: float, y_sum: float, x_squared_sum: float, y_squared_sum: float,
                                     x_tripled_sum: float, y_tripled_sum: float,
                                     length: int, xy: float, xxy: float, xyy: float) -> float:
        term0 = length * self.epsilon / 15
        term1 = (107/60 - math.log(self.epsilon)) / 2 * (x_sum + y_sum)
        term2 = (- 3/self.epsilon / 4) * (x_squared_sum + y_squared_sum + 2 * xy)
        term3 = (7/6/(self.epsilon ** 2) / 8) * (x_tripled_sum + y_tripled_sum + 3*xxy + 3*xyy)
        return term0 + term1 + term2 + term3

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
        vector1 = [probability_vector1[coord] for coord in untransmitted_coords]
        vector2 = [probability_vector2[coord] for coord in untransmitted_coords]
        vector1_squared = [probability_vector1[coord]**2 for coord in untransmitted_coords]
        vector2_squared = [probability_vector2[coord]**2 for coord in untransmitted_coords]
        xy_sketch = jl_sketch.sketch_approximations(vector1, vector2)
        xxy_sketch = jl_sketch.sketch_approximations(vector1_squared, vector2)
        xyy_sketch = jl_sketch.sketch_approximations(vector1, vector2_squared)
        for (sketch_size, xy), (_, xxy), (_, xyy) in zip(xy_sketch, xxy_sketch, xyy_sketch):
            xs.append(sketch_size)
            ys.append(known_entropy + self.calc_approximation_on_vector(x_sum, y_sum, x_squared_sum, y_squared_sum, x_tripled_sum, y_tripled_sum, len(untransmitted_coords), xy, xxy, xyy))
        plt.plot(xs, ys, color=color, label=label)


if __name__ == '__main__':
    EntropyPolyApproximation2Deg3(0.01).draw_approximation(join_create_dir('..', RESULTS_DIR_PATH, 'entropy_poly_approximation2_deg_3'))
