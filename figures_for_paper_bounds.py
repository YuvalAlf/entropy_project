import math
import os
from math import log
from random import Random
from typing import Tuple, List

import numpy as np
from matplotlib import pyplot as plt

from entropy.entropy_vec import EntropyVec
from entropy.newsgroups import NewsgroupThemeTokens
from utils.data_frame_aggragator import DataFrameAggragator
from utils.distributions import synthetic_distributions
from utils.os_utils import join_create_dir
from utils.plotting_utils import plot_horizontal, gen_plot
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42  # truetype font
mpl.rcParams['ps.fonttype'] = 42  # truetype fontfrom
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['lines.markeredgewidth'] = 0.4


mpl.rcParams.update({'font.size': 9})

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
                           entropy_vec2: EntropyVec, results_dir_path: str, add_dist: bool = True) -> None:
    assert len(entropy_vec1) == len(entropy_vec2)
    vec_length = len(entropy_vec1)

    average_vector = entropy_vec1.average_with(entropy_vec2)

    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(7.4, 2.7))
    plt.sca(ax1)
    plt.xlabel('Coordinates (Sorted)\n(a)')
    plt.ylabel('Probability Value')
    plt.title('Probability Values Histograms')
    text = ' Dist.' if add_dist else ''
    entropy_vec1.plot_histogram(color='blue', label=f'$Node_1$: {distribution1_name}{text}', alpha=0.8)
    entropy_vec2.plot_histogram(color='green', label=f'$Node_2$: {distribution2_name}{text}', alpha=0.8)
    average_vector.plot_histogram(color='red', label='Average Vector', linestyle='--', alpha=0.8)
    plt.legend()

    plt.sca(ax2)

    df_aggragator = DataFrameAggragator()
    num_samples = 500
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

    plot_horizontal(x_lims, max_entropy_value, color='black', linestyle='dashed', alpha=0.8, label='Maximum Entropy Value', clip_on=False)
    plot_horizontal(x_lims, entropy_vec1.average_with(entropy_vec2).entropy(), color='slategray', linestyle='dashdot', alpha=0.8, label="Average Vector's Entropy", clip_on=False)
    plot_horizontal(x_lims, entropy_vec1.entropy(), color='blue', linestyle='dotted', alpha=0.8, label=f'$Node_1$ Entropy ({distribution1_name})', clip_on=False)
    plot_horizontal(x_lims, entropy_vec2.entropy(), color='green', linestyle='dotted', alpha=0.8, label=f'$Node_2$ Entropy ({distribution2_name})', clip_on=False)

    plt.plot(df[TOP_K_ENTRY], df[LOWER_BOUND_VEC_1_ENTRY], color='dodgerblue', alpha=0.8, label=f'$Node_1$ Lower Bound ({distribution1_name})', clip_on=False)
    plt.plot(df[TOP_K_ENTRY], df[LOWER_BOUND_VEC_2_ENTRY], color='lawngreen', alpha=0.8, label=f'$Node_2$ Upper Bound ({distribution2_name})', clip_on=False)
    plt.plot(df[TOP_K_ENTRY], df[UPPER_BOUND_VEC_1_ENTRY], color='darkblue', alpha=0.8, label=f'$Node_1$ Lower Bound ({distribution1_name})', clip_on=False)
    plt.plot(df[TOP_K_ENTRY], df[UPPER_BOUND_VEC_2_ENTRY], color='darkgreen', alpha=0.8, label=f'$Node_2$ Upper Bound ({distribution2_name}', clip_on=False)

    plt.title('Algorithmic Bounds')
    plt.xlim((0, None))
    plt.ylim((None, max_entropy_value + 0.2))
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1.01))
    plt.xlabel('Number of Top Values Transmitted\n(b)')
    plt.ylabel('Entropy')
    plt.ticklabel_format(axis='y', useMathText='true')
    save_path = os.path.join(results_dir_path, f'{distribution1_name}_{distribution2_name}.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')


def bounds_synthetic_distributions(vector_length: int) -> None:
    distribution1_name = 'Uniform'
    entropy_vec1 = EntropyVec(np.random.uniform(low=0, high=1, size=vector_length)).normalize()
    distribution2_name = 'Beta'
    entropy_vec2 = EntropyVec(np.random.beta(a=0.2, b=100, size=vector_length)).normalize()

    result_path = join_create_dir('results', 'for_paper', 'synthetic1')
    run_entropy_simulation(distribution1_name, entropy_vec1, distribution2_name, entropy_vec2, result_path)


def bounds_newsgroups_distributions(num_newsgroups_in_each_theme: int, num_tokens: int) -> None:
    newsgroups = dict(NewsgroupThemeTokens.probability_vectors(num_newsgroups_in_each_theme, num_tokens))

    distribution1_name = 'Atheism'
    entropy_vec1 = newsgroups['alt.atheism']
    distribution2_name = 'Hockey'
    entropy_vec2 = newsgroups['rec.sport.hockey']

    result_path = join_create_dir('results', 'for_paper', 'newsgroups1')
    run_entropy_simulation(distribution1_name, entropy_vec1, distribution2_name, entropy_vec2, result_path, add_dist=False)
    # main_entropy_simulation('bounds_newsgroups', newsgroups, newsgroups)


if __name__ == '__main__':
    bounds_synthetic_distributions(vector_length=50000)
    bounds_newsgroups_distributions(num_newsgroups_in_each_theme=200, num_tokens=10000)
