import math
import os.path
from random import Random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from entropy.entropy_average_approximation import EntropyAverageApproximation
from entropy.entropy_poly_approximation import EntropyPolyApproximationDeg2
from entropy.entropy_poly_approximation1_deg3 import EntropyPolyApproximation1Deg3
from entropy.entropy_poly_approximation2 import EntropyPolyApproximation2
from entropy.entropy_poly_approximation2_deg3 import EntropyPolyApproximation2Deg3
from entropy.entropy_vec import EntropyVec
from sketches.clifford_entropy_sketch import CliffordEntropySketch
from utils.distributions import synthetic_distributions
from utils.itertools_utils import enumerate1
from utils.math_utils import calc_entropy
from utils.os_utils import join_create_dir
from utils.paths_dir import RESULTS_DIR_PATH
from utils.plotting_utils import gen_plot, plot_horizontal
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42  # truetype font
mpl.rcParams['ps.fonttype'] = 42  # truetype fontfrom
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['lines.markeredgewidth'] = 0.4

mpl.rcParams.update({'font.size': 9})


def draw_errors(epsilon: float) -> None:
    xs = np.linspace(0, epsilon, 1000)
    entropies = np.array([0 if x == 0 else -x * math.log(x) for x in xs])

    poly2 = EntropyPolyApproximationDeg2(epsilon)
    poly3 = EntropyPolyApproximation2Deg3(epsilon)
    poly2_values = np.array([poly2.calc_approximation(x) for x in xs])
    poly3_values = np.array([poly3.calc_approximation(x) for x in xs])

    with gen_plot('approximation.pdf', width=3.7, height=2.2):
        plt.plot(xs, entropies, alpha=0.8, label='Entropy Function', color='red')
        plt.ylabel('Entropy')
        plt.gca().yaxis.label.set_color('red')
        plt.gca().tick_params(axis='y', colors='red')
        plt.legend(loc='upper left', bbox_to_anchor=(0.06, 1.03))
        plt.xlabel('x')
        plt.twinx()
        plt.plot(xs, (poly2_values - entropies), alpha=0.8, label='Poly2 Error', color='springgreen')
        plt.plot(xs, (poly3_values - entropies), alpha=0.8, label='Poly3 Error', color='darkgreen')
        plt.gca().set_ylabel('Approximation Error')
        plt.gca().yaxis.label.set_color('green')
        plt.gca().tick_params(axis='y', colors='green')
        plt.ticklabel_format(scilimits=(-1, 1), useMathText=True)
        plt.legend(loc='upper left', bbox_to_anchor=(0.06, 0.87))


if __name__ == '__main__':
    draw_errors(epsilon=0.001)
