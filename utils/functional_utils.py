from itertools import islice
from typing import Callable, List, Iterable, Tuple

from utils.typing_utils import T, V, K


def map_list(func: Callable[[T], V], items: Iterable[T]) -> List[V]:
    return list(map(func, items))


def map_snd(func: Callable[[K], V], items: Iterable[Tuple[T, K]]) -> Iterable[Tuple[T, V]]:
    for item1, item2 in items:
        yield item1, func(item2)


def apply_func(func: Callable[[], V]) -> V:
    return func()


def windowed_to_start(iterable: Iterable[T]) -> Iterable[Tuple[List[T], T]]:
    values = list(iterable)
    for index, item in islice(enumerate(values), 1, len(values)):
        yield values[:index], item
