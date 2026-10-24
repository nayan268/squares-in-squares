import importlib.util
import math
import tracemalloc
import unittest
from pathlib import Path

HAS_NUMERIC=bool(importlib.util.find_spec("numpy") and importlib.util.find_spec("scipy"))

@unittest.skipUnless(HAS_NUMERIC,"optional numerical dependencies not installed")
class ScalableTests(unittest.TestCase):
    def test_direct_gradient_matches_dense_model(self):
        import numpy as np
        from squares.basin import PoseModel
        from squares.scalable import PenaltyModel
        rng=np.random.default_rng(12)
        for n in (2,5,11):
            z=np.append(rng.uniform(0.2,3,(n,3)).ravel(),3.5)
            margin,jac=PoseModel(n).evaluate(z)
            negative=np.minimum(margin,0)
            weight=400
            expected=z[-1]+weight*(negative@negative)
            gradient=2*weight*(jac.T@negative)
            gradient[-1]+=1
            actual,g=PenaltyModel(n).evaluate(z,weight)
            np.testing.assert_allclose(actual,expected,rtol=1e-12)
            np.testing.assert_allclose(g,gradient,atol=1e-8,rtol=1e-10)

    def test_thousand_square_memory_is_bounded(self):
        import numpy as np
        from squares.baseline import grid
        from squares.scalable import PenaltyModel
        pose=np.array([[s.x,s.y,s.angle] for s in grid(1000)])
        tracemalloc.start()
        model=PenaltyModel(1000)
        value,gradient=model.evaluate(np.append(pose.ravel(),32))
        _,peak=tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.assertTrue(math.isfinite(value))
        self.assertEqual(gradient.shape,(3001,))
        self.assertLess(peak,150*1024*1024)

    def test_dilation_repairs_overlap_and_rejects_coincidence(self):
        from squares.scalable import restore_feasibility
        from squares.geometry import validate
        result=restore_feasibility([[0.5,0.5,0],[1.4,0.5,0]])
        self.assertIsNotNone(result)
        self.assertGreaterEqual(validate(result[0],2),2)
        self.assertIsNone(restore_feasibility([[0,0,0],[0,0,0]]))

    def test_border_constructions_are_valid_and_improve_grids(self):
        from squares.constructions import five_square_reference
        from squares.warmstarts import add_border,extended_starts
        from squares.geometry import validate
        packing=add_border(five_square_reference())
        self.assertEqual(len(packing),10)
        self.assertLess(validate(packing,10),4)
        starts=extended_starts({5:five_square_reference()},30)
        self.assertEqual(set(starts),{5,10,17,26})
        for n,p in starts.items():
            self.assertLess(validate(p,n),math.isqrt(n-1)+1)

    def test_scalable_deadline_retains_valid_grid(self):
        from squares.scalable import solve_scalable
        from squares.geometry import validate
        packing,report=solve_scalable(1000,seconds=1e-8)
        self.assertEqual(validate(packing,1000),32)
        self.assertEqual(report["status"],"time-budget")


if __name__=="__main__":
    unittest.main()
