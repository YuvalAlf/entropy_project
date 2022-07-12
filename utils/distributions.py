from typing import Iterable, Tuple, Callable, List

import numpy as np

from entropy.entropy_vec import EntropyVec


def synthetic_distributions_generators(vector_length: int)\
        -> Iterable[Tuple[str, Callable[[], EntropyVec]]]:
    yield 'Uniform (0-0.1)', lambda: EntropyVec(np.random.uniform(0, 0.1, size=vector_length)).normalize()
    yield 'Uniform (1-2)', lambda: EntropyVec(np.random.uniform(1, 2, size=vector_length)).normalize()
    yield 'Beta a=0.1 b=100', lambda: EntropyVec(np.random.beta(a=0.1, b=100, size=vector_length)).normalize()
    yield 'Beta a=0.01 b=100', lambda: EntropyVec(np.random.beta(a=0.01, b=100, size=vector_length)).normalize()
    yield 'Exponential scale=0.02', lambda: EntropyVec(np.random.exponential(scale=0.02, size=vector_length)).normalize()
    yield 'Exponential scale=0.01', lambda: EntropyVec(np.random.exponential(scale=0.01, size=vector_length)).normalize()


def synthetic_distributions(vector_length: int) -> List[Tuple[str, EntropyVec]]:
    return [(name, distribution_generator())
            for name, distribution_generator in synthetic_distributions_generators(vector_length)]
