from typing import Iterable, Tuple

from utils.typing_utils import T, K


def enumerate1(items: Iterable[T]) -> Iterable[Tuple[int, T]]:
    yield from enumerate(items, 1)


def unzip(items: Iterable[Tuple[T, K]]) -> (Iterable[T], Iterable[K]):
    return zip(*items)


def skip(items: Iterable[Tuple[T]], items_to_skip: int) -> Iterable[T]:
    for index, item in enumerate(items):
        if index >= items_to_skip:
            yield item
