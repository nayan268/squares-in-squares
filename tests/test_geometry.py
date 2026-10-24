import math
import unittest
from squares.baseline import grid
from squares.geometry import Square, container_side, overlaps, validate


class GeometryTests(unittest.TestCase):
    def test_grids_through_1000(self):
        for n in range(1, 1001):
            self.assertEqual(validate(grid(n), n), math.isqrt(n - 1) + 1)

    def test_rotated_extent(self):
        self.assertAlmostEqual(container_side([Square(10, -20, math.pi / 4)]), math.sqrt(2))

    def test_contacts_and_overlap(self):
        origin = Square(0, 0)
        self.assertFalse(overlaps(origin, Square(1, 0)))
        self.assertFalse(overlaps(origin, Square(1, 1)))
        self.assertTrue(overlaps(origin, Square(0.999999, 0)))
        self.assertTrue(overlaps(origin, Square(0.8, 0.8, math.pi / 4)))
        self.assertFalse(overlaps(origin, Square(2, 0, math.pi / 4)))

    def test_translation_invariance(self):
        self.assertEqual(container_side(grid(7)), container_side([
            Square(s.x - 32, s.y + 100, s.angle) for s in grid(7)]))

    def test_nonfinite_rejected(self):
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                validate([Square(value, 0)])

    def test_neighbor_buckets_detect_rotated_overlap(self):
        with self.assertRaises(ValueError):
            validate([Square(1.9, 0, math.pi / 4), Square(2.1, 0)])


if __name__ == "__main__":
    unittest.main()
