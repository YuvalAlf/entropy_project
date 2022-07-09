import os.path
from random import Random

import matplotlib.pyplot as plt

from entropy.entropy_average_approximation import EntropyAverageApproximation
from entropy.entropy_poly_approximation import EntropyPolyApproximation
from sketches.clifford_entropy_sketch import CliffordEntropySketch
from utils.distributions import synthetic_distributions
from utils.os_utils import join_create_dir
from utils.paths_dir import RESULTS_DIR_PATH
from utils.plotting_utils import gen_plot, plot_horizontal


def simulate_our_sketches_for_entropy(vector_length: int, sketch_size: int) -> None:
    prng = Random(10)
    for distribution_name1, probability_vector1 in synthetic_distributions(vector_length):
        for distribution_name2, probability_vector2 in synthetic_distributions(vector_length):
            print(f'Entropy sketches for {distribution_name1} with {distribution_name2}...')
            save_dir = join_create_dir(RESULTS_DIR_PATH, 'all_sketches_comparison', f'vector_size={vector_length}', distribution_name1 + '_' + distribution_name2)
            probability_vector1.show_histogram(os.path.join(save_dir, f'distribution1_{distribution_name1}.png'))
            probability_vector2.show_histogram(os.path.join(save_dir, f'distribution2_{distribution_name2}.png'))
            average_vec = probability_vector1.average_with(probability_vector2)
            average_vec.show_histogram(os.path.join(save_dir, f'average_vector.png'))

            with gen_plot(os.path.join(save_dir, 'sketches_approximations.png'), x_label='Sketch Size', y_label='Value'):
                plot_horizontal((0, sketch_size), average_vec.entropy(), clip_on=False, color='black',
                                label='Real Entropy', linestyle='--', lw=2, zorder=0)
                for sketch_num, color in [(1, 'orange'), (2, 'red'), (3, 'yellow')]:
                    label = f'Clifford {sketch_num}'
                    sketch = CliffordEntropySketch(sketch_size, vector_length, prng.randint(0, 1000000))
                    sketch.draw(probability_vector1, probability_vector2, color, label)
                for sketch_num, color in [(1, 'green'), (2, 'lawngreen'), (3, 'olive')]:
                    label = f'Poly Approximation {sketch_num}'
                    sketch = EntropyPolyApproximation()
                    sketch.draw(sketch_size, probability_vector1, probability_vector2, color, label, prng.randint(0, 1000000))
                for epsilon, color in [(0.001, 'blue'), (0.0005, 'deepskyblue'), (0.0001, 'turquoise')]:
                    label = f'Average Approximation Epsilon={epsilon}'
                    sketch = EntropyAverageApproximation(epsilon)
                    sketch.draw(sketch_size, probability_vector1, probability_vector2, color, label, prng.randint(0, 1000000))

                plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))


if __name__ == '__main__':
    # simulate_our_sketches_for_entropy(vector_length=1000, sketch_size=500)
    # simulate_our_sketches_for_entropy(vector_length=5000, sketch_size=500)
    simulate_our_sketches_for_entropy(vector_length=10000, sketch_size=500)
