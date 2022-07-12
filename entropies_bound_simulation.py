import math
import os
from math import log
from random import Random
from typing import Tuple, List

from matplotlib import pyplot as plt

from entropy.entropy_vec import EntropyVec
from entropy.newsgroups import NewsgroupThemeTokens
from utils.data_frame_aggragator import DataFrameAggragator
from utils.distributions import synthetic_distributions
from utils.os_utils import join_create_dir
from utils.plotting_utils import plot_horizontal

COMMUNICATION_ENTRY = 'Communication'
TOP_K_ENTRY = 'Top-K transmitted'
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


def run_entropy_simulation(distribution1_name: str, entropy_vec1: EntropyVec, distribution2_name: str,
                           entropy_vec2: EntropyVec, results_dir_path: str, prng: Random) -> None:
    assert len(entropy_vec1) == len(entropy_vec2)
    vec_length = len(entropy_vec1)
    save_dir = join_create_dir(results_dir_path, f'{distribution1_name}_{distribution2_name}')
    average_vector = entropy_vec1.average_with(entropy_vec2)

    entropy_vec1.show_histogram(os.path.join(save_dir, f'{distribution1_name}_1.png'))
    entropy_vec2.show_histogram(os.path.join(save_dir, f'{distribution2_name}_2.png'))

    average_vector.show_histogram(os.path.join(save_dir, f'average_vector.png'))

    df_aggragator = DataFrameAggragator()
    num_samples = 200
    step = max(1, int(vec_length / num_samples / 2))
    for top_n in range(1, vec_length, step):
        (lower_bound1, communication1) = entropy_vec1.lower_bound(entropy_vec2, top_n)
        (upper_bound1, communication2) = entropy_vec1.upper_bound(entropy_vec2, top_n)
        (lower_bound2, communication3) = entropy_vec2.lower_bound(entropy_vec1, top_n)
        (upper_bound2, communication4) = entropy_vec2.upper_bound(entropy_vec1, top_n)
        assert len({communication1, communication2, communication2, communication4}) == 1
        df_aggragator.append_row(**{TOP_K_ENTRY: top_n,
                                    LOWER_BOUND_VEC_1_ENTRY: lower_bound1,
                                    UPPER_BOUND_VEC_1_ENTRY: upper_bound1,
                                    LOWER_BOUND_VEC_2_ENTRY: lower_bound2,
                                    UPPER_BOUND_VEC_2_ENTRY: upper_bound2})
    df = df_aggragator.to_data_frame()

    max_entropy_value = log(vec_length)
    x_lims = (0, max(vec_length, df[TOP_K_ENTRY].max()))

    plt.figure(figsize=(8, 8))
    plot_horizontal(x_lims, max_entropy_value, color='black', linestyle='dashed', alpha=0.8, label='Max Entropy Value')
    plot_horizontal(x_lims, entropy_vec1.average_with(entropy_vec2).entropy(), color='gray', linestyle='dashdot', alpha=0.8, label='Average Vector Entropy')
    plot_horizontal(x_lims, entropy_vec1.entropy(), color='deeppink', linestyle='dotted', alpha=0.8, label=f'Entropy of {distribution1_name}')
    plot_horizontal(x_lims, entropy_vec2.entropy(), color='teal', linestyle='dotted', alpha=0.8, label=f'Entropy of {distribution2_name}')

    plt.plot(df[TOP_K_ENTRY], df[LOWER_BOUND_VEC_1_ENTRY], color='blue', alpha=0.8, label=f'Lower Bound: {distribution1_name}')
    plt.plot(df[TOP_K_ENTRY], df[LOWER_BOUND_VEC_2_ENTRY], color='red', alpha=0.8, label=f'Lower Bound: {distribution2_name}')
    plt.plot(df[TOP_K_ENTRY], df[UPPER_BOUND_VEC_1_ENTRY], color='navy', alpha=0.8, label=f'Upper Bound: {distribution1_name}')
    plt.plot(df[TOP_K_ENTRY], df[UPPER_BOUND_VEC_2_ENTRY], color='firebrick', alpha=0.8, label=f'Upper Bound: {distribution2_name}')
    # for sketch_index, color in [(1, 'yellow'), (2, 'gold'), (3, 'orange'), (4, 'olive'), (5, 'goldenrod')]:
    #     sketch = CliffordEntropySketch(x_lims[1], len(entropy_vec1), prng)
    #     xs, sketch_approximations = unzip(list(sketch.sketch_approximations(average_vector))[10:])
    #
    #     plt.plot(xs, sketch_approximations, color=color, alpha=0.8, label=f'Entropy Sketch {sketch_index}', zorder=-10, lw=1)

    plt.xlim((0, None))
    plt.ylim((None, math.ceil(max_entropy_value)))
    plt.legend(loc='best')
    plt.xlabel('Number of Top Values Transmitted')
    plt.ylabel('Entropy')
    plt.savefig(os.path.join(save_dir, 'simulation.png'), dpi=300, bbox_inches='tight')
    plt.close('all')


def main_entropy_simulation(dir_name: str, distributions1: List[Tuple[str, EntropyVec]],
                            distributions2: List[Tuple[str, EntropyVec]]) -> None:
    result_path = join_create_dir('results', dir_name)
    prng = Random(10)
    for distribution1_name, entropy_vec1 in distributions1:
        for distribution2_name, entropy_vec2 in distributions2:
            print(f'Running {distribution1_name} and {distribution2_name}')
            run_entropy_simulation(distribution1_name, entropy_vec1, distribution2_name, entropy_vec2, result_path, prng)


def bounds_synthetic_distributions(vector_length: int) -> None:
    distributions1 = synthetic_distributions(vector_length)
    distributions2 = synthetic_distributions(vector_length)
    main_entropy_simulation('bounds_synthetic', distributions1, distributions2)


def bounds_newsgroups_distributions(num_newsgroups_in_each_theme: int, num_tokens: int) -> None:
    newsgroups = NewsgroupThemeTokens.probability_vectors(num_newsgroups_in_each_theme, num_tokens)
    main_entropy_simulation('bounds_newsgroups', newsgroups, newsgroups)


if __name__ == '__main__':
    bounds_synthetic_distributions(vector_length=10000)
    bounds_newsgroups_distributions(num_newsgroups_in_each_theme=100, num_tokens=5000)
