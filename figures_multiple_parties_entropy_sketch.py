import os.path
from collections import defaultdict
from random import Random

import matplotlib.pyplot as plt
import numpy as np

from entropy.entropy_average_approximation import EntropyAverageApproximation
from entropy.entropy_poly_approximation import EntropyPolyApproximationDeg2
from entropy.entropy_poly_approximation1_deg3 import EntropyPolyApproximation1Deg3
from entropy.entropy_poly_approximation2 import EntropyPolyApproximation2
from entropy.entropy_poly_approximation2_deg3 import EntropyPolyApproximation2Deg3
from entropy.entropy_vec import EntropyVec
from sketches.clifford_entropy_sketch import CliffordEntropySketch
from utils.distributions import synthetic_distributions
from utils.functional_utils import map_list
from utils.itertools_utils import enumerate1
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


def simulate_sketches_synthetic_std(vector_length: int, max_sketch_size: int, max_num_vectors: int) -> None:
    prng = Random(20)

    for num_vectors in [200]:
        print(num_vectors)
        plt.figure()
        vectors = [EntropyVec(np.random.uniform(0.0, vector_length, size=vector_length)).normalize() for _ in range(num_vectors)]
        epsilon = max(value + 0.0000000001 for vec in vectors for value in vec)
        poly2 = EntropyPolyApproximationDeg2(epsilon)
        clifford_values = defaultdict(list)
        clifford = CliffordEntropySketch(max_sketch_size, vector_length, prng.randint(0, 1000000))
        for sketch_size, sketches in clifford.sketch_approximations_multiple(vectors):
            if sketch_size > 10:
                clifford_values[sketch_size].append(clifford.sketch_calculation_multiple(sketches))

        poly2_values = defaultdict(list)
        for sketch_size, sketch_value in poly2.sketch_approximations_multiple(max_sketch_size, vectors, prng.randint(0, 1000000)):
            poly2_values[sketch_size].append(sketch_value)

        # plt.plot(list(clifford_values.keys()), map_list(np.std, clifford_values.values()), color='green', label='Clifford', alpha=0.85)
        plt.plot(list(clifford_values.keys()), clifford_values.values(), label=f'CC, {num_vectors} Parties', alpha=0.85)
        plt.plot(list(poly2_values.keys()), poly2_values.values(), label=f'Poly2, {num_vectors} Parties', alpha=0.85)
        x_lims = (min(poly2_values.keys()), max(poly2_values.keys()))
        average_vec = EntropyVec.average_of(vectors)
        plot_horizontal(x_lims, average_vec.entropy(), color='black', linestyle='dashed', alpha=0.8,
                        label='Max Entropy Value')

        plt.legend(loc='best')
        plt.xlabel('Sketch Size')
        plt.ylabel('Entropy')
        plt.xlim((0, None))
        plt.ylim((0, None))
        plt.ticklabel_format(axis='y', useMathText='true')
        plt.tight_layout()
        save_dir = join_create_dir('results', 'for_paper', 'sketches_multiple')
        plt.savefig(os.path.join(save_dir, f'{num_vectors}.png'), dpi=300, bbox_inches='tight')
        plt.close('all')


if __name__ == '__main__':
    simulate_sketches_synthetic_std(vector_length=10000, max_sketch_size=501, max_num_vectors=20)
