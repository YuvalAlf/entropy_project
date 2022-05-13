from typing import Callable, List, Iterable

from utils.typing_utils import T, V


def map_list(func: Callable[[T], V], items: Iterable[T]) -> List[V]:
    return list(map(func, items))

