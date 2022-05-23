from typing import Iterable, Tuple

from utils.typing_utils import T


def combinations_with_repetitions(items: Iterable[T]) -> Iterable[Tuple[T, T]]:
    list_items = list(items)
    for i in range(len(list_items)):
        for j in range(i, len(list_items)):
            yield list_items[i], list_items[j]
