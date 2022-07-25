import os.path
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


def plot_sketches(distribution1_name: str, entropy_vec1: EntropyVec, distribution2_name: str, entropy_vec2: EntropyVec,
                  max_sketch_size: int, save_path: str) -> None:
    prng = Random(20)
    vector_length = len(entropy_vec1)
    average_vector = entropy_vec1.average_with(entropy_vec2)

    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(7.0, 2.7))
    plt.sca(ax1)
    plt.xlabel('Coordinates (Sorted)\n(a)')
    plt.ylabel('Probability Value')
    plt.title('Probability Values Histograms')
    entropy_vec1.plot_histogram(color='blue', label=f'$Node_1$: {distribution1_name} Dist.', alpha=0.8)
    entropy_vec2.plot_histogram(color='green', label=f'$Node_2$: {distribution2_name} Dist.', alpha=0.8)
    average_vector.plot_histogram(color='red', label='Average Vector', linestyle='--', alpha=0.8)
    plt.legend()
    plt.ylim((0, 0.00025))

    plt.sca(ax2)

    min_communication = float('inf')

    for sketch_num, color in [(1, 'gold'), (2, 'orange'), (3, 'red')]:
        label = f'CC ({sketch_num})'
        print(label)
        sketch = CliffordEntropySketch(max_sketch_size, vector_length, prng.randint(0, 1000000))
        min_communication = min(min_communication, sketch.draw_communication(entropy_vec1, entropy_vec2, color, label, alpha=0.8))

    for sketch_num, (epsilon, color) in enumerate1([(0.0002, 'lightseagreen'), (0.0002, 'royalblue'), (0.0002, 'darkturquoise')]):
        label = f'Poly2 ({sketch_num})'
        print(label)
        sketch = EntropyPolyApproximationDeg2(epsilon)
        min_communication = min(min_communication,
                                sketch.draw_communication(max_sketch_size, entropy_vec1, entropy_vec2, color, label,
                                                          prng.randint(0, 1000000)))
    # for epsilon, color in [(0.0002, 'teal')]:
    #     label = f'Poly3 $\\varepsilon$={epsilon}'
    #     sketch = EntropyPolyApproximation2Deg3(epsilon)
    #     min_communication = min(min_communication, sketch.draw_communication(max_sketch_size, entropy_vec1, entropy_vec2, color, label, prng.randint(0, 1000000)))
    # for sketch_num, (epsilon, color) in enumerate1([(0.0002, 'lightseagreen'), (0.0002, 'royalblue'), (0.0002, 'darkturquoise')]):
    #     label = f'Poly2 ({sketch_num})'
    #     sketch = EntropyPolyApproximation2Deg3(epsilon)
    #     min_communication = min(min_communication, sketch.draw_communication(max_sketch_size, entropy_vec1, entropy_vec2, color, label, prng.randint(0, 1000000)))
        # for sketch_num, color in [(1, 'green'), (2, 'lawngreen'), (3, 'olive')]:
        #     label = f'Poly Approximation {sketch_num}'
        #     sketch = EntropyPolyApproximation()
        #     sketch.draw(sketch_size, probability_vector1, probability_vector2, color, label, prng.randint(0, 1000000))
        # for sketch_num, color in [(1, 'violet'), (2, 'darkviolet'), (3, 'slateblue')]:
        #     label = f'Poly Approximation2 {sketch_num}'
        #     sketch = EntropyPolyApproximation2()
        #     sketch.draw(sketch_size, probability_vector1, probability_vector2, color, label, prng.randint(0, 1000000))
        # for epsilon, color in [(0.001, 'blue'), (0.0005, 'deepskyblue'), (0.0001, 'turquoise')]:
        #     label = f'Average Approximation Epsilon={epsilon}'
        #     sketch = EntropyAverageApproximation(epsilon)
        #     sketch.draw(sketch_size, probability_vector1, probability_vector2, color, label, prng.randint(0, 1000000))
        # for epsilon, color in [(0.0003, 'deepskyblue'), (0.0002, 'cyan'), (0.0001, 'turquoise')]:
        #     label = f'Poly3 eps={epsilon} no free var'
        #     sketch = EntropyPolyApproximation1Deg3(epsilon)
        #     sketch.draw(sketch_size, probability_vector1, probability_vector2, color, label, prng.randint(0, 1000000))

    plot_horizontal((0, min_communication), average_vector.entropy(), color='black',
                    label='Real Entropy', linestyle='--', zorder=0)
    plt.xlim((0, min_communication))

    plt.title('Sketches Approximation')
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1.01))
    plt.xlabel('Communication Cost\n(b)')
    plt.ylabel('Entropy')
    plt.ticklabel_format(axis='y', useMathText='true')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')


def simulate_sketches_synthetic(vector_length: int, max_sketch_size: int) -> None:
    distribution1_name = 'Uniform (1)'
    entropy_vec1 = EntropyVec(np.random.uniform(0.01, 1, size=vector_length)).normalize()
    distribution2_name = 'Uniform (2)'
    entropy_vec2 = EntropyVec(np.random.uniform(1, 2, size=vector_length)).normalize()

    result_dir_path = join_create_dir('results', 'for_paper', 'sketches')
    save_path = os.path.join(result_dir_path, f'{distribution1_name}_{distribution2_name}.pdf')
    plot_sketches(distribution1_name, entropy_vec1, distribution2_name, entropy_vec2, max_sketch_size, save_path)


if __name__ == '__main__':
    # simulate_our_sketches_for_entropy(vector_length=1000, sketch_size=500)
    # simulate_our_sketches_for_entropy(vector_length=5000, sketch_size=500)
    simulate_sketches_synthetic(vector_length=10000, max_sketch_size=230)
    # simulate_sketches_synthetic(vector_length=10000, max_sketch_size=105)
