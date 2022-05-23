import numpy as np
from typing import Iterable, Tuple, Callable

from entropy.entropy_vec import EntropyVec
from utils.combinatorics_utils import combinations_with_repetitions

TOP_N_ENTRY = 'Top-N'
DISTRIBUTION_1_ENTRY = 'Distribution 1'
DISTRIBUTION_2_ENTRY = 'Distribution 2'
LOWER_BOUND_VEC_1_ENTRY = 'Lower Bound Vec1'
UPPER_BOUND_VEC_1_ENTRY = 'Upper Bound Vec1'
LOWER_BOUND_VEC_2_ENTRY = 'Lower Bound Vec2'
UPPER_BOUND_VEC_2_ENTRY = 'Upper Bound Vec2'


def generators() -> Iterable[Tuple[str, Callable[[], float]]]:
    np.random.seed(200)
    yield 'Uniform (0-1)', lambda: np.random.uniform(0, 1)
    # yield 'Uniform (1-2)', lambda: np.random.uniform(1, 2)
    # yield 'Binomial n=100 p=0.5', lambda: np.random.binomial(n=100, p=0.5)
    # yield 'Binomial n=100 p=0.8', lambda: np.random.binomial(n=100, p=0.8)
    # yield 'Exponential scale=1', lambda: np.random.exponential(scale=1)


def main() -> None:
    vec_length = 1000
    print(f'{TOP_N_ENTRY},{DISTRIBUTION_1_ENTRY},{DISTRIBUTION_2_ENTRY},'
          f'{LOWER_BOUND_VEC_1_ENTRY},{UPPER_BOUND_VEC_1_ENTRY},{LOWER_BOUND_VEC_2_ENTRY},{UPPER_BOUND_VEC_2_ENTRY}')
    for (generator_name1, generator1), (generator_name2, generator2) in combinations_with_repetitions(generators()):
        entropy_vec1 = EntropyVec.gen_rand(vec_length, generator1)
        entropy_vec2 = EntropyVec.gen_rand(vec_length, generator2)
        print(entropy_vec1.average_with(entropy_vec2).entropy())
        for top_n in range(vec_length):
            lower_bound_vec1 = entropy_vec1.lower_bound(entropy_vec2, top_n)
            upper_bound_vec1 = entropy_vec1.upper_bound(entropy_vec2, top_n)
            lower_bound_vec2 = entropy_vec2.lower_bound(entropy_vec1, top_n)
            upper_bound_vec2 = entropy_vec2.upper_bound(entropy_vec1, top_n)
            print(f'{top_n},{generator_name1},{generator_name2},'
                  f'{lower_bound_vec1},{upper_bound_vec1},{lower_bound_vec2},{upper_bound_vec2}')


if __name__ == '__main__':
    main()
