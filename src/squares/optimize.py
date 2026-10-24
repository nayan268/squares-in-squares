"""Small-n multistart squeeze/relax solver using optional NumPy and SciPy."""
import math
import time
from dataclasses import dataclass, asdict
from .baseline import grid
from .geometry import Square, validate


@dataclass
class SolverConfig:
    seed: int = 0
    restarts: int = 8
    maxiter: int = 250
    squeeze_steps: int = 12
    shrink: float = 0.025
    clearance: float = 1e-7


def _libraries():
    try:
        import numpy as np
        from scipy.optimize import minimize
    except ImportError as error:
        raise RuntimeError("Install solver dependencies: pip install -e '.[optimize]'") from error
    return np, minimize


def margins(vector, n, side, clearance=0.0):
    """Nonnegative margins mean wall containment and pairwise SAT separation."""
    np, _ = _libraries()
    a = np.asarray(vector).reshape(n, 3)
    x, y, theta = a.T
    half = (np.abs(np.cos(theta)) + np.abs(np.sin(theta))) / 2
    walls = np.concatenate((x-half, y-half, side-x-half, side-y-half)) - clearance
    i, j = np.triu_indices(n, 1)
    dx, dy = x[j]-x[i], y[j]-y[i]
    delta = theta[i]-theta[j]
    radius = (1 + np.abs(np.cos(delta)) + np.abs(np.sin(delta))) / 2
    axes = np.stack((theta[i], theta[i]+math.pi/2,
                     theta[j], theta[j]+math.pi/2), axis=1)
    gaps = np.abs(dx[:, None]*np.cos(axes) + dy[:, None]*np.sin(axes)) - radius[:, None]
    return np.concatenate((walls, np.max(gaps, axis=1)-clearance))


def _packing(vector, n):
    np, _ = _libraries()
    return [Square(float(x), float(y), float(t % (math.pi/2)))
            for x, y, t in np.asarray(vector).reshape(n, 3)]


def _safe_candidate(vector, n, clearance):
    """Expand centers slightly, then independently screen actual square geometry."""
    np, _ = _libraries()
    a = np.asarray(vector).reshape(n, 3).copy()
    a[:, :2] *= 1 + clearance
    result = _packing(a.ravel(), n)
    try:
        side = validate(result, n)
    except ValueError:
        return None
    return result, side


class _TimeBudgetExpired(Exception):
    pass


def solve(n, config=None, initial=None, time_limit=None):
    np, scipy_minimize = _libraries()
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit <= 0):
        raise ValueError("time_limit must be finite and positive")
    config = config or SolverConfig()
    if not 1 <= n <= 1000 or isinstance(n, bool) or not isinstance(n, int):
        raise ValueError("n must be an integer in 1..1000")
    if config.restarts < 1 or config.maxiter < 1 or config.squeeze_steps < 0:
        raise ValueError("restarts/maxiter must be positive; squeeze_steps nonnegative")
    if not (0 < config.shrink < 0.5) or not (0 < config.clearance < 0.01):
        raise ValueError("invalid shrink or clearance")
    incumbent = list(initial) if initial is not None else grid(n)
    best_side = validate(incumbent, n)
    start_side = best_side
    start = time.perf_counter()
    history = []
    status = "complete"
    deadline = start + time_limit if time_limit is not None else None
    def minimize(fun, *args, **kwargs):
        def guarded(v):
            if deadline is not None and time.perf_counter() >= deadline:
                raise _TimeBudgetExpired()
            return fun(v)
        return scipy_minimize(guarded, *args, **kwargs)
    rng = np.random.default_rng(config.seed)
    # Area bound certifies perfect-square grids; don't waste optimization time.
    if abs(best_side - math.sqrt(n)) < 1e-12:
        return incumbent, {"n": n, "initial_L": start_side, "best_L": best_side,
                           "seconds": time.perf_counter()-start, "config": asdict(config),
                           "history": [], "status": "area-bound"}
    width = math.isqrt(n-1)+1
    try:
        for restart in range(config.restarts):
            side = start_side
            if restart == 0:
                a = np.array([[s.x,s.y,s.angle] for s in incumbent])
            elif restart % 3 == 1:
                # Randomly select occupied lattice cells, including different holes.
                cells = rng.choice(width*width, n, replace=False)
                a = np.column_stack((cells % width+0.5, cells // width+0.5,
                                     rng.uniform(-0.6,0.6,n)))
                a[:,:2] += rng.normal(0,0.12,(n,2))
            elif restart % 3 == 2:
                a = np.column_stack((rng.uniform(0.5,side-0.5,(n,2)),
                                     rng.uniform(-math.pi/4,math.pi/4,n)))
            else:
                a = np.array([[s.x,s.y,s.angle] for s in incumbent])
                side = best_side
                a[:,:2] += rng.normal(0,0.10,(n,2))
                a[:,2] += rng.normal(0,0.2,n)
            vector = a.ravel()
            step = config.shrink
            attempts = []
            for iteration in range(config.squeeze_steps):
                target = max(math.sqrt(n)+config.clearance, side-step)
                bounds = [(0,target),(0,target),(-math.pi,math.pi)] * n
                trial = vector.copy().reshape(n,3)
                trial[:,:2] *= target/side
                def penalty(v):
                    violation = np.minimum(margins(v,n,target,config.clearance),0)
                    return float(violation @ violation)
                relaxed = minimize(penalty, trial.ravel(), method="L-BFGS-B", bounds=bounds,
                                   options={"maxiter": min(config.maxiter,100),
                                            "ftol":1e-14,"gtol":1e-8})
                feasible = minimize(lambda v: 0.0, relaxed.x, method="SLSQP",
                                    bounds=bounds,
                                    constraints={"type":"ineq","fun":lambda v:
                                                 margins(v,n,target,config.clearance)},
                                    options={"maxiter":config.maxiter,"ftol":1e-11})
                gap = float(np.min(margins(feasible.x,n,target,config.clearance)))
                candidate = _safe_candidate(feasible.x,n,config.clearance)
                accepted = candidate is not None and gap >= -1e-9
                attempts.append({"target_L":target,"min_margin":gap,"accepted":accepted})
                if accepted:
                    vector, side = feasible.x, target
                    if candidate[1] < best_side:
                        incumbent, best_side = candidate
                else:
                    # Retain the relaxed configuration on the first failed step so
                    # constrained polishing can escape an infeasible random start.
                    if iteration == 0:
                        vector = relaxed.x
                    step *= 0.5
                if step < 1e-5:
                    break
            # Constrained polishing makes L a variable. The same margins apply.
            z = np.append(vector, side)
            polished = minimize(lambda v: v[-1], z, method="SLSQP",
                                bounds=[(0,start_side),(0,start_side),(-math.pi,math.pi)]*n
                                       +[(math.sqrt(n),start_side)],
                                constraints={"type":"ineq","fun":lambda v:
                                             margins(v[:-1],n,v[-1],config.clearance)},
                                options={"maxiter":config.maxiter,"ftol":1e-11})
            candidate = _safe_candidate(polished.x[:-1],n,config.clearance)
            feasible_polish = (candidate is not None and
                              np.min(margins(polished.x[:-1],n,polished.x[-1])) >= -1e-9)
            if feasible_polish and candidate[1] < best_side:
                incumbent, best_side = candidate
            history.append({"restart":restart,"best_L":best_side,
                            "polish_success":bool(polished.success),
                            "polish_valid":bool(feasible_polish), "attempts":attempts,
                            "seconds":time.perf_counter()-start})
    except _TimeBudgetExpired:
        status = "time-budget"
    return incumbent, {"n":n,"initial_L":start_side,"best_L":best_side,
                       "seconds":time.perf_counter()-start,"config":asdict(config),
                       "history":history,"status":status,"time_limit":time_limit}
