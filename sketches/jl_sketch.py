from random import Random
from typing import List

from sketches.projection_sketch import ProjectionSketch
from utils.math_utils import inner_product


class JohnsonLindenstraussSketch(ProjectionSketch):
    def gen_projection_vector(self, prng: Random) -> List[float]:
        return [prng.gauss(0, 1) for _ in range(self.vector_size)]

    def sketch_calculation(self, sketch1: List[float], sketch2: List[float]) -> float:
        return inner_product(sketch1, sketch2) / len(sketch1)
