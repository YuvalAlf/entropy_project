import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from utils.functional_utils import map_list
from utils.os_utils import join_create_dir
from utils.paths_dir import RESULTS_DIR_PATH
from utils.plotting_utils import gen_plot


@dataclass
class EntropyPolyApproximation:
    epsilon: float = 0.001

    def calc_approximation(self, x: float) -> float:
        return -1250 * x * x + 7.991088 * x + 0.000125

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


EntropyPolyApproximation().draw_approximation(join_create_dir('..', RESULTS_DIR_PATH, 'entropy_poly_approximation'))
