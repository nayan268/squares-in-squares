import math
import unittest
import importlib.util
HAS_NUMERIC = bool(importlib.util.find_spec("numpy") and importlib.util.find_spec("scipy"))
if HAS_NUMERIC:
    import numpy as np
from squares.basin import PoseModel,BasinConfig,solve_basin,metropolis,propose
from squares.optimize import margins
from squares.baseline import grid
from squares.geometry import validate


@unittest.skipUnless(HAS_NUMERIC,"optional solver dependencies not installed")
class BasinTests(unittest.TestCase):
    def test_analytic_jacobian_matches_finite_differences(self):
        rng=np.random.default_rng(91)
        for n in (2,5,11):
            model=PoseModel(n)
            z=np.append(rng.uniform(0.2,2.8,(n,3)).ravel(),4.1)
            values,jac=model.evaluate(z)
            step=1e-6
            eye=np.eye(len(z))*step
            numeric=np.column_stack([(model.evaluate(z+d)[0]-model.evaluate(z-d)[0])/(2*step) for d in eye])
            np.testing.assert_allclose(jac,numeric,atol=2e-6,rtol=2e-6)
            np.testing.assert_allclose(values,margins(z[:-1],n,z[-1],1e-7),atol=2e-14)

    def test_metropolis_acceptance(self):
        self.assertTrue(metropolis(4,3.9,0,0.99))
        self.assertFalse(metropolis(4,4.1,0,0))
        self.assertTrue(metropolis(4,4.01,0.1,0.1))
        self.assertFalse(metropolis(4,4.1,0.01,0.9))

    def test_proposals_are_repeatable_and_move_groups(self):
        for hop in range(6):
            a,label=propose(grid(11),np.random.default_rng(4),hop,4.25)
            b,label2=propose(grid(11),np.random.default_rng(4),hop,4.25)
            np.testing.assert_array_equal(a,b)
            self.assertEqual(label,label2)
            self.assertGreater(np.sum(np.any(a != np.array([[s.x,s.y,s.angle] for s in grid(11)]),axis=1)),1)

    def test_deadline_preserves_valid_incumbent(self):
        packing,report=solve_basin(11,BasinConfig(seconds=1e-8,hops=2))
        self.assertEqual(validate(packing,11),4)
        self.assertEqual(report["status"],"time-budget")
        self.assertEqual(report["completed_hops"],0)

    def test_soft_walk_never_exports_an_invalid_state(self):
        from unittest.mock import patch
        invalid=np.zeros(33)
        diagnostic={"local_success":False,"min_margin":-1.0,
                    "walk_vector":invalid,"walk_energy":0.0}
        with patch("squares.basin.quench",return_value=(None,diagnostic)):
            packing,report=solve_basin(11,BasinConfig(seconds=1,hops=1))
        self.assertEqual(validate(packing,11),4)
        self.assertTrue(report["history"][0]["accepted"])
        self.assertIsNone(report["history"][0]["current_L"])

    def test_legacy_deadline_preserves_valid_incumbent(self):
        from squares.optimize import solve,SolverConfig
        packing,report=solve(11,SolverConfig(restarts=100),time_limit=0.001)
        self.assertEqual(validate(packing,11),4)
        self.assertEqual(report["status"],"time-budget")

    @unittest.skipUnless(importlib.util.find_spec("PIL"),"PNG plotting dependency not installed")
    def test_comparison_updates_progress_and_resumes(self):
        import tempfile,json
        from pathlib import Path
        from squares.basin_experiment import compare
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/"comparison"
            first=compare(output,n=4,seeds=(1,2),seconds=0.05)
            progress=json.loads((output/"progress.json").read_text())
            self.assertEqual(len(progress["runs"]),4)
            resumed=compare(output,n=4,seeds=(1,2),seconds=0.05,resume=True)
            self.assertEqual(first["summary"],resumed["summary"])
            with self.assertRaises(ValueError):
                compare(output,n=4,seeds=(1,2),seconds=0.1,resume=True)

    def test_area_bound_and_configuration_checks(self):
        packing,report=solve_basin(4,BasinConfig(seconds=0.1))
        self.assertEqual(validate(packing,4),2)
        self.assertEqual(report["status"],"area-bound")
        with self.assertRaises(ValueError):
            solve_basin(11,BasinConfig(seconds=-1))


if __name__=="__main__":
    unittest.main()
