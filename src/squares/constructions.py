"""Reference constructions; never injected into blind optimizer benchmarks."""
import math
from .geometry import Square, validate


def five_square_reference(clearance=1e-8):
    """Four corner squares plus a 45-degree center square.

    The mathematical limit is 2 + 1/sqrt(2). A small positive expansion of
    center positions prevents decimal rounding from creating tiny overlaps.
    """
    if not math.isfinite(clearance) or clearance <= 0:
        raise ValueError("clearance must be finite and positive")
    side = 2 + 1 / math.sqrt(2)
    squares = [Square(x, y) for x, y in
               ((0.5, 0.5), (side - 0.5, 0.5),
                (side - 0.5, side - 0.5), (0.5, side - 0.5))]
    squares.append(Square(side / 2, side / 2, math.pi / 4))
    squares = [Square(s.x * (1 + clearance), s.y * (1 + clearance), s.angle)
               for s in squares]
    validate(squares, 5)
    return squares
