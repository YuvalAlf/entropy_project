import os.path
from random import Random

from matplotlib import pyplot as plt

from sketches.jl_sketch import JohnsonLindenstraussSketch
from utils.distributions import synthetic_distributions
from utils.math_utils import inner_product
from utils.os_utils import join_create_dir
from utils.paths_dir import RESULTS_DIR_PATH
from utils.plotting_utils import plot_horizontal, gen_plot


def inner_product_sketch_visualization(vector_length: int, sketch_size: int) -> None:
    prng = Random(10)
    for distribution1_name, probability_vector1 in synthetic_distributions(vector_length):
        for distribution2_name, probability_vector2 in synthetic_distributions(vector_length):
            print(f'Running {distribution1_name} and {distribution2_name}')
            save_dir = join_create_dir(RESULTS_DIR_PATH, 'jl_sketch', f'{distribution1_name}_{distribution2_name}')
            probability_vector1.show_histogram(os.path.join(save_dir, f'{distribution1_name}_1.png'))
            probability_vector2.show_histogram(os.path.join(save_dir, f'{distribution2_name}_2.png'))

            with gen_plot(os.path.join(save_dir, 'sketch_approximation.png'), x_label='Sketch Size', y_label='Value'):
                plot_horizontal((0, sketch_size), inner_product(probability_vector1, probability_vector2), clip_on=False,
                                color='black', label='Inner Product Value', linestyle='--', lw=2, zorder=0)
                for sketch_num, color in [(1, 'orange'), (2, 'red'), (3, 'yellow')]:
                    label = f'JL Approximation {sketch_num}'
                    sketch = JohnsonLindenstraussSketch(sketch_size, vector_length, prng.randint(0, 10000))
                    sketch.draw(probability_vector1, probability_vector2, color, label)
                plt.legend()


if __name__ == '__main__':
    inner_product_sketch_visualization(vector_length=10000, sketch_size=2000)
