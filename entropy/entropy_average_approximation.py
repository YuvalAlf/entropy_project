import math
import os
from dataclasses import dataclass

import numpy as np
import seaborn as sns

from utils.plotting_utils import gen_plot


@dataclass
class EntropyAverageApproximation:
    epsilon: float

    def calc_average(self, x: float, y: float) -> float:
        ln_2 = math.log(2)
        ln_eps = math.log(self.epsilon)
        x_term = -(32 * ln_2 - 21) / self.epsilon * x * x / 4 + (61 / 10 * ln_2 - 451 / 120 - ln_eps / 2) * x
        y_term = -(32 * ln_2 - 21) / self.epsilon * y * y / 4 + (61 / 10 * ln_2 - 451 / 120 - ln_eps / 2) * y
        x_y_term = 3 / 10 * (16 * ln_2 - 13) / self.epsilon * x * y
        constant_term = -self.epsilon * (128 * ln_2 - 99) / 60
        return x_term + y_term + x_y_term + constant_term

    def calc_on_vector(self, sum_x_squared: float, sum_x: float, sum_y_squared: float, sum_y: float, x_dot_y: float) -> float:
        ln_2 = math.log(2)
        ln_eps = math.log(self.epsilon)
        x_term = -(32 * ln_2 - 21) / self.epsilon * sum_x_squared / 4 + (61 / 10 * ln_2 - 451 / 120 - ln_eps / 2) * sum_x
        y_term = -(32 * ln_2 - 21) / self.epsilon * sum_y_squared / 4 + (61 / 10 * ln_2 - 451 / 120 - ln_eps / 2) * sum_y
        x_y_term = 3 / 10 * (16 * ln_2 - 13) / self.epsilon * x_dot_y
        constant_term = -self.epsilon * (128 * ln_2 - 99) / 60
        return x_term + y_term + x_y_term + constant_term

    def real_average(self, x: float, y: float) -> float:
        average = (x + y) / 2
        if average == 0:
            return 0
        return -average * math.log(average)

    def draw_approximation(self, save_dir: float):
        num_samples = 80
        xs_ys = list(np.linspace(0, self.epsilon, num_samples + 1))
        xs_ys_labels = ['0' if x == 0 else f'{x:1.1e}' if i % 10 == 0 else '' for i, x in enumerate(xs_ys)]
        error_values = [[self.calc_average(x, y) - self.real_average(x, y) for x in xs_ys] for y in xs_ys]
        with gen_plot(os.path.join(save_dir, f'epsilon={self.epsilon:1.1e}.png'),
                      width=8, height=6.5, x_label='X Values', y_label='Y Values',
                      title=f'Entropy Approximation, Epsilon={self.epsilon:1.1E}'):
            sns.heatmap(error_values, cmap='YlGnBu', linewidth=0, annot=False,
                        yticklabels=xs_ys_labels, xticklabels=xs_ys_labels).invert_yaxis()

