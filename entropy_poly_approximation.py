import numpy as np

from entropy.entropy_approximation import EntropyAverageApproximation
from utils.os_utils import join_create_dir
from utils.paths_dir import RESULTS_DIR_PATH


def draw_figures() -> None:
    save_dir = join_create_dir(RESULTS_DIR_PATH, 'entropy_approximation')
    for epsilon in np.linspace(0.00001, 0.001, 20):
        print(f'Epsilon = {epsilon}')
        EntropyAverageApproximation(epsilon).draw_approximation(save_dir)


if __name__ == '__main__':
    draw_figures()
