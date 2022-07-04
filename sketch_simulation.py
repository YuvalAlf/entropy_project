import math
import os
from math import log
from random import Random
from typing import Iterable, Tuple, Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from entropy.entropy_sketch import EntropySketch
from entropy.entropy_vec import EntropyVec
from newsgroups_entropy import NewsgroupThemeTokens
from utils.combinatorics_utils import combinations_with_repetitions, combinations_without_repetitions
from utils.data_frame_aggragator import DataFrameAggragator
from utils.functional_utils import map_list
from utils.math_utils import list_average
from utils.os_utils import join_create_dir, encode_json
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
    print(f'Running {distribution1_name} and {distribution2_name}, Vector Length = {vec_length}...')
    save_dir = join_create_dir(results_dir_path, f'{distribution1_name}_{distribution2_name}')

    entropy_vec1.show_histogram(os.path.join(save_dir, f'{distribution1_name}_1.png'))
    entropy_vec2.show_histogram(os.path.join(save_dir, f'{distribution2_name}_2.png'))
    entropy_vec1.average_with(entropy_vec2).show_histogram(os.path.join(save_dir, f'average_vector.png'))

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
    for sketch_index, color in [(1, 'yellow'), (2, 'gold'), (3, 'orange'), (4, 'olive'), (5, 'goldenrod')]:
        max_x = x_lims[1]
        xs = list(range(1, max_x + 1))
        sketch = EntropySketch(max_x, len(entropy_vec1), prng)
        sketch_approximations = list(sketch.sketch_approximations(list_average(entropy_vec1, entropy_vec2)))

        plt.plot(xs, sketch_approximations, color=color, alpha=0.8, label=f'Entropy Sketch {sketch_index}', zorder=-10, lw=1)

    plt.xlim((0, None))
    plt.ylim((None, math.ceil(max_entropy_value)))
    plt.legend(loc='best')
    plt.xlabel('Number of Top Values Transmitted')
    plt.ylabel('Entropy')
    plt.savefig(os.path.join(save_dir, 'simulation.png'), dpi=300, bbox_inches='tight')
    plt.close('all')


def sketch_simulation(vector_size: str) -> None:
    result_path = join_create_dir('results', 'sketch_simulation')
    prng = Random(100)
    exp_vec = EntropyVec(np.random.beta(a=0.001, b=100, size=vector_size)).normalize()
    uniform_vec = EntropyVec(np.random.uniform(100, 101, size=vector_size)).normalize()
    sketch = EntropySketch(vector_size, vector_size, prng)

    weight_to_vec = dict()
    for weight in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        average_vec = EntropyVec([weight*a + (1-weight)*b for a, b in zip(exp_vec, uniform_vec)])
        average_vec.show_histogram(os.path.join(result_path, f'{weight}.png'))
        average_vec_entropy = average_vec.entropy()
        print(average_vec_entropy)
        weight_to_vec[weight] = [abs(app - average_vec_entropy) for app in sketch.sketch_approximations(sketch.apply(average_vec)[1])]
    plt.figure(figsize=(8, 8))
    for weight, approx in weight_to_vec.items():
        xs = list(range(1, vector_size+1))
        plt.plot(xs[250:], approx[250:], label=str(weight), alpha=0.7, lw=1)

    plt.legend(loc='best')
    plt.xlabel('Sketch Size')
    plt.ylabel('Entropy Approximation Error')
    plt.savefig(os.path.join(result_path, 'simulation.png'), dpi=300, bbox_inches='tight')
    plt.close('all')


if __name__ == '__main__':
    sketch_simulation(2000)
