"""Float geometry for fast local screening; not an interval certificate."""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Square:
    x: float
    y: float
    angle: float = 0.0  # Radians, counterclockwise.

    def axes(self):
        c, s = math.cos(self.angle), math.sin(self.angle)
        return ((c, s), (-s, c))

    def corners(self):
        u, v = self.axes()
        return tuple((self.x + a * u[0] + b * v[0],
                      self.y + a * u[1] + b * v[1])
                     for a, b in ((-0.5, -0.5), (0.5, -0.5),
                                  (0.5, 0.5), (-0.5, 0.5)))


def container_side(squares):
    if not squares:
        raise ValueError("packing must not be empty")
    points = [p for square in squares for p in square.corners()]
    xs, ys = zip(*points)
    return max(max(xs) - min(xs), max(ys) - min(ys))


def overlaps(a, b, tolerance=0.0):
    """Separating-axis test allowing boundary contact.

    Positive tolerance permits projected penetration. The default is zero.
    Near-contact floats must be checked with the official metric.
    """
    if a.angle == 0.0 and b.angle == 0.0:
        return abs(a.x - b.x) < 1.0 - tolerance and abs(a.y - b.y) < 1.0 - tolerance
    au, av = a.axes()
    bu, bv = b.axes()
    dx, dy = b.x - a.x, b.y - a.y
    for ux, uy in (au, av, bu, bv):
        distance = abs(dx * ux + dy * uy)
        radius = 0.5 * sum(abs(v[0] * ux + v[1] * uy)
                           for v in (au, av, bu, bv))
        if distance >= radius - tolerance:
            return False
    return True


def validate(squares, expected_n=None, tolerance=0.0):
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if not squares or (expected_n is not None and len(squares) != expected_n):
        raise ValueError("incorrect number of squares")
    for i, square in enumerate(squares):
        if not all(math.isfinite(v) for v in (square.x, square.y, square.angle)):
            raise ValueError("non-finite coordinate at square {}".format(i))
    # Overlapping unit squares have center distance <= sqrt(2). Width-2
    # buckets therefore need only their eight immediate neighbors.
    buckets = {}
    for i, square in enumerate(squares):
        cell = (math.floor(square.x / 2), math.floor(square.y / 2))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in buckets.get((cell[0] + dx, cell[1] + dy), ()):
                    if overlaps(squares[j], square, tolerance):
                        raise ValueError("squares {} and {} overlap".format(j, i))
        buckets.setdefault(cell, []).append(i)
    side = container_side(squares)
    if not math.isfinite(side):
        raise ValueError("non-finite container side")
    return side
