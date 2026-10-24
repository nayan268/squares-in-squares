import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from squares.baseline import grid
from squares.constructions import five_square_reference
from squares.geometry import Square, overlaps, validate
from squares.plot import plot_packings


class ReferenceTests(unittest.TestCase):
    def test_five_square_reference(self):
        packing=five_square_reference()
        side=validate(packing,5)
        self.assertGreater(side,2+1/math.sqrt(2))
        self.assertLess(side,2+1/math.sqrt(2)+1e-6)

    def test_svg_highlights_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"overlap.svg"
            plot_packings([("Bad packing",[Square(0,0),Square(0.5,0)])],path)
            source=path.read_text()
            self.assertIn("OVERLAP",source)
            self.assertIn("#ee6a70",source)
            with self.assertRaises(FileExistsError):
                plot_packings([("Grid",grid(4))],path)


@unittest.skipUnless(importlib.util.find_spec("scipy") and importlib.util.find_spec("numpy"),
                     "optional optimizer dependencies not installed")
class OptimizationTests(unittest.TestCase):
    def test_sat_margins_agree_with_independent_geometry(self):
        import numpy as np
        from squares.optimize import margins
        rng=np.random.default_rng(456)
        for _ in range(100):
            data=rng.uniform(-2,2,(2,3))
            a,b=(Square(*row) for row in data)
            pair_margin=margins(data.ravel(),2,10)[-1]
            self.assertEqual(pair_margin<0,overlaps(a,b))

    def test_area_bound_control(self):
        from squares.optimize import solve,SolverConfig
        packing,report=solve(4,SolverConfig(seed=3,restarts=1,maxiter=20))
        self.assertEqual(validate(packing,4),2)
        self.assertEqual(report["status"],"area-bound")

    def test_solver_never_replaces_with_invalid_or_worse(self):
        from squares.optimize import solve,SolverConfig
        packing,report=solve(5,SolverConfig(seed=1,restarts=1,maxiter=30,squeeze_steps=1))
        self.assertLessEqual(validate(packing,5),3)
        self.assertEqual(report["best_L"],validate(packing,5))


if __name__=="__main__":
    unittest.main()
