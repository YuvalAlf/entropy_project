import os.path
from random import Random

import matplotlib.pyplot as plt

from entropy.distributions import synthetic_distributions
from entropy.entropy_sketch import EntropySketch
from utils.itertools_utils import unzip
from utils.os_utils import join_create_dir
from utils.paths_dir import RESULTS_DIR_PATH
from utils.plotting_utils import gen_plot


def entropy_sketch_visualization() -> None:
    prng = Random(10)
    vector_length = 2000
    max_sketch_size = 500
    num_sketches = 4
    min_sketch_size = 10
    for distribution_name, probability_vector in synthetic_distributions(vector_length):
        save_dir = join_create_dir(RESULTS_DIR_PATH, 'entropy_sketch_visualization', distribution_name)
        probability_vector.show_histogram(os.path.join(save_dir, 'distribution.png'))
        with gen_plot(os.path.join(save_dir, 'sketch_approximation.png'), x_label='Sketch Size', y_label='Entropy'):
            for sketch_num in range(1, num_sketches + 1):
                sketch = EntropySketch(max_sketch_size, vector_length, prng)
                sketch_sizes, approximation_values = unzip(list(sketch.sketch_approximations(probability_vector))[min_sketch_size:])
                plt.scatter(sketch_sizes, approximation_values, clip_on=False, alpha=0.6,
                            label=f'Entropy Sketch Approximation {sketch_num}', s=2, zorder=1)
            plt.plot(sketch_sizes, [probability_vector.entropy()] * len(sketch_sizes), clip_on=False, color='black',
                     label='Real Entropy', linestyle='--', lw=2, zorder=0)
            plt.legend(loc='lower right')


if __name__ == '__main__':
    entropy_sketch_visualization()
