from abc import abstractmethod, ABC
from collections import Iterable
from dataclasses import dataclass
from random import Random
from typing import List, Tuple

import matplotlib.pyplot as plt

from utils.itertools_utils import unzip, enumerate1
from utils.math_utils import inner_product


@dataclass
class ProjectionSketch(ABC):
    sketch_size: int
    vector_size: int
    random_seed: int

    @abstractmethod
    def gen_projection_vector(self, prng: Random) -> List[float]:
        pass

    @abstractmethod
    def sketch_calculation(self, sketch1: List[float], sketch2: List[float]) -> float:
        pass

    def projection_vectors(self) -> Iterable[List[float]]:
        prng = Random(self.random_seed)
        for _ in range(self.sketch_size):
            yield self.gen_projection_vector(prng)

    def project(self, vector: List[float]) -> List[float]:
        return [inner_product(projection_vector, vector) for projection_vector in self.projection_vectors()]

    def sketch_approximations(self, vector1: List[float], vector2: List[float]) -> Iterable[Tuple[int, float]]:
        sketch1, sketch2 = [], []
        for projection_vector in self.projection_vectors():
            sketch1.append(inner_product(vector1, projection_vector))
            sketch2.append(inner_product(vector2, projection_vector))
        for sketch_size in range(20, len(sketch1)):
            yield sketch_size, self.sketch_calculation(sketch1[:sketch_size], sketch2[:sketch_size])

    def sketch_approximations_multiple(self, vectors: List[List[float]]) -> Iterable[Tuple[int, List[float]]]:
        sketches = [[] for _ in vectors]
        for sketch_size, projection_vector in enumerate1(self.projection_vectors()):
            for sketch, vec in zip(sketches, vectors):
                sketch.append(inner_product(vec, projection_vector))
            yield sketch_size, sketches

    def draw(self, vector1: List[float], vector2: List[float], color: str, label: str, **kwargs: str) -> None:
        xs, ys = unzip(list(self.sketch_approximations(vector1, vector2))[10:])
        plt.plot(xs, ys, color=color, label=label, **kwargs)
