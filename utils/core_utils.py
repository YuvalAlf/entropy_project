from typing import Tuple, Any

from utils.typing_utils import T


def fst(arr: Tuple[T, ...]) -> T:
    return arr[0]


def snd(arr: Any) -> T:
    return arr[1]
