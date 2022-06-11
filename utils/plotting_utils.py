from typing import Tuple

from matplotlib import pyplot as plt


def plot_horizontal(x_lims: Tuple[float, float], y_value: float, **plt_kwargs) -> None:
    plt.plot(x_lims, (y_value, y_value), **plt_kwargs)







