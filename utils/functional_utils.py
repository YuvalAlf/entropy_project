from itertools import islice
from typing import Callable, List, Iterable, Tuple

from utils.typing_utils import T, V


def map_list(func: Callable[[T], V], items: Iterable[T]) -> List[V]:
    return list(map(func, items))


def windowed_to_start(iterable: Iterable[T]) -> Iterable[Tuple[List[T], T]]:
    values = list(iterable)
    for index, item in islice(enumerate(values), 1, len(values)):
        yield values[:index], item
