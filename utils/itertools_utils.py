from typing import Iterable, Tuple

from utils.typing_utils import T


def enumerate1(items: Iterable[T]) -> Iterable[Tuple[int, T]]:
    yield from enumerate(items, 1)
