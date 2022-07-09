import os.path
from random import Random

from utils.distributions import synthetic_distributions
from inner_product.inner_product_sketch import InnerProductSketch
from utils.os_utils import join_create_dir
from utils.paths_dir import RESULTS_DIR_PATH


def inner_product_sketch_visualization(vector_length: int, sketch_size: int) -> None:
    prng = Random(10)
    for distribution1_name, probability_vector1 in synthetic_distributions(vector_length):
        for distribution2_name, probability_vector2 in synthetic_distributions(vector_length):
            print(f'Running {distribution1_name} and {distribution2_name}')
            save_dir = join_create_dir(RESULTS_DIR_PATH, 'jl_sketch', f'{distribution1_name}_{distribution2_name}')
            probability_vector1.show_histogram(os.path.join(save_dir, f'{distribution1_name}_1.png'))
            probability_vector2.show_histogram(os.path.join(save_dir, f'{distribution2_name}_2.png'))
            InnerProductSketch(sketch_size, vector_length, prng).draw_approximation(probability_vector1, probability_vector2, os.path.join(save_dir, 'JL.png'))


if __name__ == '__main__':
    inner_product_sketch_visualization(vector_length=20000, sketch_size=2000)
