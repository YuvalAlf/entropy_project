from collections import Iterable
from dataclasses import dataclass
from functools import cached_property
from random import Random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray, cumsum

from utils.itertools_utils import enumerate1, unzip
from utils.math_utils import gen_matrix, inner_product
from utils.plotting_utils import gen_plot, plot_horizontal


@dataclass
class InnerProductSketch:
    sketch_size: int
    vector_size: int
    prng: Random

    @cached_property
    def used_seed(self) -> int:
        return self.prng.randint(0, 1000000)

    def projection_vectors(self) -> Iterable[List[float]]:
        random_generator = Random(self.used_seed)
        for _ in range(self.sketch_size):
            yield [random_generator.gauss(0, 1) for _ in range(self.vector_size)]

    def project(self, vector: List[float]) -> Iterable[float]:
        for projection_vector in self.projection_vectors():
            yield inner_product(projection_vector, vector)

    def apply(self, vector1: List[float], vector2: List[float]) -> float:
        return inner_product(self.project(vector1), self.project(vector2)) / self.sketch_size

    def sketch_approximations(self, vector1: List[float], vector2: List[float]) -> Iterable[Tuple[int, float]]:
        projection_values = []
        for projection_vector in self.projection_vectors():
            projection1 = inner_product(vector1, projection_vector)
            projection2 = inner_product(vector2, projection_vector)
            projection_values.append(projection1 * projection2)
        for index, sum_projections in enumerate1(cumsum(projection_values)):
            yield index, sum_projections / index

    def draw_approximation(self, vector1: List[float], vector2: List[float], save_path: str) -> None:
        with gen_plot(save_path):
            plot_horizontal((0, self.sketch_size), inner_product(vector1, vector2), color='red', label='Inner Product', linestyle='--')
            xs, ys = unzip(list(self.sketch_approximations(vector1, vector2))[10:])
            plt.plot(xs, ys, color='green', label=f'JL')
            plt.legend()
