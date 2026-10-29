import importlib.util
import math
import unittest

HAS_NUMERIC=bool(importlib.util.find_spec("numpy") and importlib.util.find_spec("scipy"))
if HAS_NUMERIC:
    import numpy as np


@unittest.skipUnless(HAS_NUMERIC, "optional numerical dependencies not installed")
class StructuredTests(unittest.TestCase):
    def test_inflation_jacobian_matches_finite_differences(self):
        from squares.structured import InflationModel
        rng=np.random.default_rng(32)
        for n in (2, 5, 11):
            model=InflationModel(n)
            z=np.append(rng.uniform(0.15, 0.85, (n, 3)).ravel(), 0.18)
            values, jac=model.evaluate(z)
            step=1e-6
            eye=np.eye(len(z))*step
            numeric=np.column_stack([(model.evaluate(z+d)[0]-model.evaluate(z-d)[0])/(2*step)
                                     for d in eye])
            np.testing.assert_allclose(jac, numeric, atol=2e-6, rtol=2e-6)
            self.assertEqual(values.shape[0], 4*n+n*(n-1)//2)

    def test_maximal_scale_recovers_grid(self):
        from squares.baseline import grid
        from squares.structured import packing_pose, maximal_scale, _candidate
        from squares.geometry import validate
        for n in (2, 5, 11, 29):
            pose=packing_pose(grid(n))
            expected=1/(math.isqrt(n-1)+1)
            self.assertAlmostEqual(maximal_scale(pose, 0), expected, places=12)
            candidate=_candidate(pose, 1e-9)
            self.assertIsNotNone(candidate)
            self.assertGreaterEqual(validate(candidate[0], n), math.sqrt(n))

    def test_templates_are_repeatable_and_distinct(self):
        from squares.structured import template_pose
        kinds=("diagonal-1", "diagonal-2", "central-block", "staggered-rows",
               "tilted-strips", "grid-jitter")
        poses=[]
        for kind in kinds:
            a=template_pose(19, kind, np.random.default_rng(8))
            b=template_pose(19, kind, np.random.default_rng(8))
            np.testing.assert_array_equal(a, b)
            poses.append(a)
        self.assertTrue(all(np.any(poses[0]!=pose) for pose in poses[1:]))

    def test_deadline_retains_valid_incumbent(self):
        from squares.structured import solve_structured, StructuredConfig
        from squares.baseline import grid
        from squares.geometry import validate
        packing, report=solve_structured(11, StructuredConfig(seconds=1e-8, trials=2))
        self.assertEqual(validate(packing, 11), validate(grid(11), 11))
        self.assertEqual(report["status"], "time-budget")

    def test_published_research_seeds_are_valid_and_better(self):
        from squares.research_seeds import available, published_seed
        from squares.baseline import grid
        from squares.geometry import validate
        expected={11:3.877084,17:4.675531,18:4.822876,19:4.885619,29:5.933834}
        for n in available():
            side=validate(published_seed(n),n)
            self.assertLess(side,validate(grid(n),n))
            self.assertLess(side,expected[n]+1e-5)

    def test_balanced_row_families_preserve_count_and_nonoverlap(self):
        from squares.family import balanced_row_counts, family_packing, family_side
        from squares.geometry import validate
        for n in (28, 31, 39, 84):
            for variant in range(4):
                counts=balanced_row_counts(n,variant,np.random.default_rng(7))
                self.assertEqual(sum(counts),n)
                parameters=np.zeros(len(counts)+1)
                packing=family_packing(counts,parameters)
                self.assertAlmostEqual(validate(packing,n),family_side(counts,parameters),places=6)

    def test_family_deadline_retains_incumbent(self):
        from squares.baseline import grid
        from squares.family import FamilyConfig, solve_family
        from squares.geometry import validate
        packing,report=solve_family(28,FamilyConfig(seconds=1e-8,trials=2))
        self.assertEqual(validate(packing,28),validate(grid(28),28))
        self.assertEqual(report["status"],"time-budget")

    def test_interlocked_family_pose_has_expected_shape(self):
        from squares.family import balanced_row_counts, interlocked_pose
        counts=balanced_row_counts(31)
        parameters=np.zeros(len(counts)+4)
        parameters[2:4]=1/len(counts)
        pose=interlocked_pose(counts,parameters)
        self.assertEqual(pose.shape,(31,3))
        self.assertTrue(np.all(np.isfinite(pose)))

    def test_research_border_family_preserves_square_offset(self):
        from squares.geometry import validate
        from squares.research_seeds import propagated_seed
        expected=(28,39,52,67,84)
        previous=4.885619
        for n in expected:
            packing=propagated_seed(n)
            side=validate(packing,n)
            self.assertLess(side,math.isqrt(n-1)+1)
            self.assertGreater(side,previous)
            previous=side
        with self.assertRaises(KeyError):
            propagated_seed(31)


if __name__=="__main__":
    unittest.main()
