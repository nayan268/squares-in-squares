"""Deterministic, exactly representable grid baseline."""
import math
from .geometry import Square


def grid(n: int):
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 1000:
        raise ValueError("n must be an integer from 1 through 1000")
    width = math.isqrt(n - 1) + 1
    return [Square(i % width + 0.5, i // width + 0.5, 0.0) for i in range(n)]
