import os.path
from random import Random

import matplotlib.pyplot as plt

from entropy.distributions import synthetic_distributions
from entropy.entropy_sketch import CliffordEntropySketch
from utils.itertools_utils import unzip
from utils.os_utils import join_create_dir
from utils.paths_dir import RESULTS_DIR_PATH
from utils.plotting_utils import gen_plot, plot_horizontal


def simulate_johnson_lindenshtraus(vector_length: int, max_sketch_size: int) -> None:
    prng = Random(10)
    for distribution1_name, probability_vector1 in synthetic_distributions(vector_length):
        for distribution2_name, probability_vector2 in synthetic_distributions(vector_length):
            print(f'Running {distribution1_name} and {distribution2_name}')
            save_dir = join_create_dir(RESULTS_DIR_PATH, 'jl_approx', f'{distribution1_name}_{distribution2_name}')
            average_vec = probability_vector1.average_with(probability_vector2)
            probability_vector1.show_histogram(os.path.join(save_dir, f'{distribution1_name}_1.png'))
            probability_vector2.show_histogram(os.path.join(save_dir, f'{distribution2_name}_2.png'))
            average_vec.show_histogram(os.path.join(save_dir, f'average_vec.png'))
            with gen_plot(os.path.join(save_dir, f'simulation.png')):
                plot_horizontal((0, max_sketch_size), average_vec.entropy(), label='Average Vector Entropy', color='r')
                plt.plot(*unzip(CliffordEntropySketch(max_sketch_size, vector_length, prng).sketch_approximations(average_vec)),
                         label='Entropy Sketch', color='b')
                for epsilon in [0.01, 0.001, 0.0001]:
                    jl_sketch = JlEntropySketch(epsilon, max_sketch_size, vector_length)
                    plt.plot(*unzip(jl_sketch.sketch_approximations(probability_vector1, probability_vector2)),
                             label='Entropy Sketch', color='b')
                plt.legend()


if __name__ == '__main__':
    simulate_johnson_lindenshtraus(vector_length=10000, max_sketch_size=500)
