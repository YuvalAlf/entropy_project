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
from utils.itertools_utils import enumerate1
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


mpl.rcParams.update({'font.size': 8})

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


def bounds_synthetic_distributions_danny(vec_length: int) -> None:

    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(7.3, 2.4))

    plt.sca(ax1)

    distribution1_name = 'Beta'
    entropy_vec1 = EntropyVec(np.random.beta(a=0.1, b=100, size=vec_length)).normalize()
    distribution2_name = 'Uniform'
    entropy_vec2 = EntropyVec(np.random.uniform(low=0, high=1, size=vec_length)).normalize()

    df_aggragator = DataFrameAggragator()
    num_samples = 500
    step = max(1, int(vec_length / num_samples / 2))
    for i, top_n in enumerate1(range(1, vec_length, step)):
        print(f'{i / num_samples * 50: .2f}%')
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

    plot_horizontal(x_lims, max_entropy_value, color='black', linestyle='dashed', alpha=0.8, label='Maximum Entropy Value')
    plot_horizontal(x_lims, entropy_vec1.average_with(entropy_vec2).entropy(), color='slategray', linestyle='dashdot', alpha=0.8, label="Average Vector's Entropy")
    plot_horizontal(x_lims, entropy_vec1.entropy(), color='blue', linestyle='dotted', alpha=0.8, label=f'$Node_1$ Entropy ({distribution1_name})')
    plot_horizontal(x_lims, entropy_vec2.entropy(), color='green', linestyle='dotted', alpha=0.8, label=f'$Node_2$ Entropy ({distribution2_name})')

    plt.plot(df[TOP_K_ENTRY], df[LOWER_BOUND_VEC_1_ENTRY], color='dodgerblue', alpha=0.8, label=f'$Node_1$ L. Bound ({distribution1_name})')
    plt.plot(df[TOP_K_ENTRY], df[LOWER_BOUND_VEC_2_ENTRY], color='lawngreen', alpha=0.8, label=f'$Node_2$ U. Bound ({distribution2_name})')
    plt.plot(df[TOP_K_ENTRY], df[UPPER_BOUND_VEC_1_ENTRY], color='darkblue', alpha=0.8, label=f'$Node_1$ L. Bound ({distribution1_name})')
    plt.plot(df[TOP_K_ENTRY], df[UPPER_BOUND_VEC_2_ENTRY], color='darkgreen', alpha=0.8, label=f'$Node_2$ U. Bound ({distribution2_name})')

    plt.title('Uniform Dist. vs Beta Dist.')
    plt.xlim((0, None))
    plt.ylim((None, max_entropy_value + 0.1))
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1.01))
    plt.xlabel('Number of Top Values Transmitted\n(a)')
    plt.ylabel('Entropy')
    plt.ticklabel_format(axis='y', useMathText='true')

    plt.sca(ax2)

    distribution1_name = 'Beta1'
    entropy_vec1 = EntropyVec(np.random.beta(a=0.1, b=100, size=vec_length)).normalize()
    distribution2_name = 'Beta2'
    entropy_vec2 = EntropyVec(np.random.beta(a=0.02, b=100, size=vec_length)).normalize()

    df_aggragator = DataFrameAggragator()
    num_samples = 500
    step = max(1, int(vec_length / num_samples / 2))
    for i, top_n in enumerate1(range(1, vec_length, step)):
        print(f'{i / num_samples * 50: .2f}%')
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

    plot_horizontal(x_lims, max_entropy_value, color='black', linestyle='dashed', alpha=0.8, label='Maximum Entropy Value')
    plot_horizontal(x_lims, entropy_vec1.average_with(entropy_vec2).entropy(), color='slategray', linestyle='dashdot', alpha=0.8, label="Average Vector's Entropy")
    plot_horizontal(x_lims, entropy_vec1.entropy(), color='blue', linestyle='dotted', alpha=0.8, label=f'$Node_1$ Entropy ({distribution1_name})')
    plot_horizontal(x_lims, entropy_vec2.entropy(), color='green', linestyle='dotted', alpha=0.8, label=f'$Node_2$ Entropy ({distribution2_name})')

    plt.plot(df[TOP_K_ENTRY], df[LOWER_BOUND_VEC_1_ENTRY], color='dodgerblue', alpha=0.8, label=f'$Node_1$ L. Bound ({distribution1_name})')
    plt.plot(df[TOP_K_ENTRY], df[LOWER_BOUND_VEC_2_ENTRY], color='lawngreen', alpha=0.8, label=f'$Node_2$ U. Bound ({distribution2_name})')
    plt.plot(df[TOP_K_ENTRY], df[UPPER_BOUND_VEC_1_ENTRY], color='darkblue', alpha=0.8, label=f'$Node_1$ L. Bound ({distribution1_name})')
    plt.plot(df[TOP_K_ENTRY], df[UPPER_BOUND_VEC_2_ENTRY], color='darkgreen', alpha=0.8, label=f'$Node_2$ U. Bound ({distribution2_name})')

    plt.title('Two Beta Distributions')
    plt.xlim((0, None))
    plt.ylim((None, max_entropy_value + 0.1))
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1.01))
    plt.xlabel('Number of Top Values Transmitted\n(b)')
    plt.ylabel('Entropy')
    plt.ticklabel_format(axis='y', useMathText='true')


    save_dir = join_create_dir('results', 'for_paper', 'sketch_motivation')
    save_path = os.path.join(save_dir, f'sketch_motivation.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')


if __name__ == '__main__':
    bounds_synthetic_distributions_danny(vec_length=20000)
