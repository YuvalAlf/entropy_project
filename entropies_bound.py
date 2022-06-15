import math
import os
from math import log

import numpy as np
from typing import Iterable, Tuple, Callable

from matplotlib import pyplot as plt

from entropy.entropy_vec import EntropyVec
from newsgroups_entropy import NewsgroupThemeTokens
from utils.combinatorics_utils import combinations_with_repetitions, combinations_without_repetitions
from utils.data_frame_aggragator import DataFrameAggragator
from utils.functional_utils import map_snd, apply_func
from utils.os_utils import join_create_dir, encode_json
from utils.plotting_utils import plot_horizontal

TOP_N_ENTRY = 'Top-N'
DISTRIBUTION_1_ENTRY = 'Distribution 1'
DISTRIBUTION_2_ENTRY = 'Distribution 2'
AVERAGE_ENTROPY_ENTRY = 'Average Entropy'
LOWER_BOUND_VEC_1_ENTRY = 'Lower Bound Vec1'
UPPER_BOUND_VEC_1_ENTRY = 'Upper Bound Vec1'
LOWER_BOUND_VEC_2_ENTRY = 'Lower Bound Vec2'
UPPER_BOUND_VEC_2_ENTRY = 'Upper Bound Vec2'
ENTROPY_1_ENTRY = 'Vec1 Entropy'
ENTROPY_2_ENTRY = 'Vec2 Entropy'
VEC_LENGTH_ENTRY = 'Vector Length'


def synthetic_distributions(vector_size: int) -> Iterable[Tuple[str, Callable[[], EntropyVec]]]:
    np.random.seed(200)
    yield 'Beta a=0.1 b=100', lambda: EntropyVec(np.random.beta(a=0.1, b=100, size=vector_size)).normalize()
    yield 'Beta a=0.01 b=100', lambda: EntropyVec(np.random.beta(a=0.01, b=100, size=vector_size)).normalize()
    yield 'Uniform (0-0.1)', lambda: EntropyVec(np.random.uniform(0, 0.1, size=vector_size)).normalize()
    yield 'Uniform (1-2)', lambda: EntropyVec(np.random.uniform(1, 2, size=vector_size)).normalize()
    yield 'Exponential scale=0.02', lambda: EntropyVec(np.random.exponential(scale=0.02, size=vector_size)).normalize()
    yield 'Exponential scale=0.01', lambda: EntropyVec(np.random.exponential(scale=0.01, size=vector_size)).normalize()


def run_entropy_simulation(distribution1_name: str, entropy_vec1: EntropyVec, distribution2_name: str,
                           entropy_vec2: EntropyVec, results_dir_path: str) -> None:
    assert len(entropy_vec1) == len(entropy_vec2)
    vec_length = len(entropy_vec1)
    print(f'Running {distribution1_name} and {distribution2_name}, Vector Length = {vec_length}...')
    save_dir = join_create_dir(results_dir_path, f'{distribution1_name}_{distribution2_name}')

    entropy_vec1.show_histogram(os.path.join(save_dir, f'{distribution1_name}_1.png'))
    entropy_vec2.show_histogram(os.path.join(save_dir, f'{distribution2_name}_2.png'))

    df_aggragator = DataFrameAggragator()
    num_samples = 100
    step = max(1, int(vec_length / num_samples / 2))
    for top_n in range(0, vec_length // 2, step):
        df_aggragator.append_row(**{TOP_N_ENTRY: top_n,
                                    LOWER_BOUND_VEC_1_ENTRY: entropy_vec1.lower_bound(entropy_vec2, top_n),
                                    UPPER_BOUND_VEC_1_ENTRY: entropy_vec1.upper_bound(entropy_vec2, top_n),
                                    LOWER_BOUND_VEC_2_ENTRY: entropy_vec2.lower_bound(entropy_vec1, top_n),
                                    UPPER_BOUND_VEC_2_ENTRY: entropy_vec2.upper_bound(entropy_vec1, top_n)})
    df = df_aggragator.to_data_frame()
    max_entropy_value = log(vec_length)
    x_lims = (0, vec_length)

    plt.figure(figsize=(8, 8))
    plot_horizontal(x_lims, max_entropy_value, color='black', linestyle='dashed', alpha=0.8, label='Max Entropy Value')
    plot_horizontal(x_lims, entropy_vec1.average_with(entropy_vec2).entropy(), color='gray', linestyle='dashdot', alpha=0.8, label='Average Vector Entropy')
    plot_horizontal(x_lims, entropy_vec1.entropy(), color='deeppink', linestyle='dotted', alpha=0.8, label=f'Entropy of {distribution1_name}')
    plot_horizontal(x_lims, entropy_vec2.entropy(), color='teal', linestyle='dotted', alpha=0.8, label=f'Entropy of {distribution2_name}')

    plt.plot(df[TOP_N_ENTRY], df[LOWER_BOUND_VEC_1_ENTRY], color='blue', alpha=0.8, label=f'Lower Bound: {distribution1_name}')
    plt.plot(df[TOP_N_ENTRY], df[LOWER_BOUND_VEC_2_ENTRY], color='red', alpha=0.8, label=f'Lower Bound: {distribution2_name}')
    plt.plot(df[TOP_N_ENTRY], df[UPPER_BOUND_VEC_1_ENTRY], color='navy', alpha=0.8, label=f'Upper Bound: {distribution1_name}')
    plt.plot(df[TOP_N_ENTRY], df[UPPER_BOUND_VEC_2_ENTRY], color='firebrick', alpha=0.8, label=f'Upper Bound: {distribution2_name}')

    plt.xlim((0, None))
    plt.ylim((None, math.ceil(max_entropy_value)))
    plt.legend(loc='lower right')
    plt.xlabel('Number of Top Values Transmitted')
    plt.ylabel('Entropy')
    plt.savefig(os.path.join(save_dir, 'simulation.png'), dpi=300, bbox_inches='tight')
    plt.close('all')


def main_entropy_simulation(dir_name: str, distributions: Iterable[Tuple[Tuple[str, EntropyVec], Tuple[str, EntropyVec]]]) -> None:
    result_path = join_create_dir('results', dir_name)

    for (distribution1_name, entropy_vec1), (distribution2_name, entropy_vec2) in distributions:
        run_entropy_simulation(distribution1_name, entropy_vec1, distribution2_name, entropy_vec2, result_path)


def main_synthetic_distributions(vector_length: int) -> None:
    distributions = combinations_with_repetitions(synthetic_distributions(vector_length))
    applied_distributions = (((name1, dist1()), (name2, dist2())) for (name1, dist1), (name2, dist2) in distributions)
    main_entropy_simulation('synthetic', applied_distributions)


def main_newsgroups_distributions(num_newgroups_to_fetch: int, num_tokens: int) -> None:
    newsgroups = NewsgroupThemeTokens.fetch(num_newgroups_to_fetch)
    unique_tokens, token_to_location = NewsgroupThemeTokens.unique_tokens(newsgroups, num_tokens)
    themes_and_entropy_vecs = [(newsgroup.theme, EntropyVec(newsgroup.probability_vector(unique_tokens)))
                            for newsgroup in newsgroups]
    dir_path = join_create_dir('results', 'newsgroups')
    encode_json(unique_tokens, os.path.join(dir_path, 'top_tokens.json'))
    main_entropy_simulation('newsgroups', combinations_without_repetitions(themes_and_entropy_vecs))


if __name__ == '__main__':
    main_synthetic_distributions(vector_length=10000)
    main_newsgroups_distributions(num_newgroups_to_fetch=100, num_tokens=10000)
