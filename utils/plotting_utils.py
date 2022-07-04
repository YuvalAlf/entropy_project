import contextlib
from typing import Tuple, ContextManager

from matplotlib import pyplot as plt


def plot_horizontal(x_lims: Tuple[float, float], y_value: float, **plt_kwargs) -> None:
    plt.plot(x_lims, (y_value, y_value), **plt_kwargs)


@contextlib.contextmanager
def gen_plot(save_path: str, width: float, height: float, x_label: str = '', y_label: str = '', title: str = '') -> ContextManager[None]:
    plt.figure(figsize=(width, height))
    yield
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')





