import os.path
from random import Random

import matplotlib.pyplot as plt

from sketches.clifford_entropy_sketch import CliffordEntropySketch
from utils.distributions import synthetic_distributions
from utils.os_utils import join_create_dir
from utils.paths_dir import RESULTS_DIR_PATH
from utils.plotting_utils import gen_plot, plot_horizontal


def entropy_sketch_visualization() -> None:
    prng = Random(10)
    vector_length = 2000
    max_sketch_size = 1000
    for distribution_name, probability_vector in synthetic_distributions(vector_length):
        print(f'Visualizing entropy sketch for {distribution_name}...')
        save_dir = join_create_dir(RESULTS_DIR_PATH, 'entropy_sketch_visualization', distribution_name)
        probability_vector.show_histogram(os.path.join(save_dir, 'distribution.png'))
        with gen_plot(os.path.join(save_dir, 'sketch_approximation.png'), x_label='Sketch Size', y_label='Entropy'):
            plot_horizontal((0, max_sketch_size), probability_vector.entropy(), clip_on=False, color='black',
                            label='Real Entropy', linestyle='--', lw=2, zorder=0)
            for sketch_num, color in [(1, 'orange'), (2, 'red'), (3, 'yellow')]:
                label = f'Entropy Sketch Approximation {sketch_num}'
                sketch = CliffordEntropySketch(max_sketch_size, vector_length, prng.randint(0, 10000))
                sketch.draw(probability_vector, probability_vector, color, label)
            plt.legend()


if __name__ == '__main__':
    entropy_sketch_visualization()
