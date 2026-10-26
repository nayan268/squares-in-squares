"""Structured maximal-inflation search for small and medium puzzles.

The search works in a unit container and maximizes the common square side.
It combines maximal inflation with reusable geometric templates instead of
perturbing every square independently.
"""
import math
import time
from dataclasses import asdict, dataclass

from .baseline import grid
from .basin import BudgetExpired, metropolis
from .geometry import Square, validate
from .optimize import _libraries


@dataclass
class StructuredConfig:
    seed: int = 0
    seconds: float = 30.0
    trials: int = 1000
    maxiter: int = 120
    temperature: float = 0.015
    clearance: float = 1e-8


class InflationModel:
    """Wall/SAT margins for equal squares inflated inside a unit box."""

    def __init__(self, n, clearance=1e-8, deadline=None):
        self.n = n
        self.clearance = clearance
        self.deadline = deadline
        self.np, _ = _libraries()
        self.i, self.j = self.np.triu_indices(n, 1)
        self.calls = 0

    def evaluate(self, z):
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise BudgetExpired()
        self.calls += 1
        np, n = self.np, self.n
        pose = np.asarray(z[:-1]).reshape(n, 3)
        x, y, theta = pose.T
        size = float(z[-1])
        c, s = np.cos(theta), np.sin(theta)
        half = (np.abs(c) + np.abs(s)) / 2
        dh = (-np.sign(c) * s + np.sign(s) * c) / 2
        walls = np.concatenate((x-size*half, y-size*half,
                                1-x-size*half, 1-y-size*half))
        jac = np.zeros((4*n+len(self.i), 3*n+1))
        r = np.arange(n)
        jac[r, 3*r] = 1
        jac[n+r, 3*r+1] = 1
        jac[2*n+r, 3*r] = -1
        jac[3*n+r, 3*r+1] = -1
        for k in range(4):
            jac[k*n+r, 3*r+2] = -size*dh
            jac[k*n+r, -1] = -half

        i, j = self.i, self.j
        delta = theta[i]-theta[j]
        cd, sd = np.cos(delta), np.sin(delta)
        radius = (1+np.abs(cd)+np.abs(sd))/2
        dr = (-np.sign(cd)*sd+np.sign(sd)*cd)/2
        dx, dy = x[j]-x[i], y[j]-y[i]
        angles = np.stack((theta[i], theta[i]+math.pi/2,
                           theta[j], theta[j]+math.pi/2), axis=1)
        ca, sa = np.cos(angles), np.sin(angles)
        dot = dx[:, None]*ca + dy[:, None]*sa
        gaps = np.abs(dot)-size*radius[:, None]
        k = np.argmax(gaps, axis=1)
        q = np.arange(len(i))
        co, si = ca[q, k], sa[q, k]
        sign = np.sign(dot[q, k])
        rotation = sign*(-dx*si+dy*co)
        rr = 4*n+q
        jac[rr, 3*i] = -sign*co
        jac[rr, 3*i+1] = -sign*si
        jac[rr, 3*j] = sign*co
        jac[rr, 3*j+1] = sign*si
        jac[rr, 3*i+2] = -size*dr + np.where(k < 2, rotation, 0)
        jac[rr, 3*j+2] = size*dr + np.where(k >= 2, rotation, 0)
        jac[rr, -1] = -radius
        return np.concatenate((walls, gaps[q, k]))-self.clearance, jac


def packing_pose(packing):
    """Normalize a valid unit-square packing into a unit container."""
    np, _ = _libraries()
    points = [p for square in packing for p in square.corners()]
    xmin = min(p[0] for p in points)
    ymin = min(p[1] for p in points)
    side = validate(packing)
    return np.array([[(s.x-xmin)/side, (s.y-ymin)/side, s.angle]
                     for s in packing], dtype=float)


def maximal_scale(pose, clearance=1e-10):
    """Largest common side feasible for fixed centers and orientations."""
    np, _ = _libraries()
    pose = np.asarray(pose, dtype=float)
    if pose.ndim != 2 or pose.shape[1] != 3 or len(pose) == 0:
        raise ValueError("pose must have shape (n,3)")
    x, y, theta = pose.T
    c, s = np.cos(theta), np.sin(theta)
    half = (np.abs(c)+np.abs(s))/2
    limits = [float(np.min(x/half)), float(np.min(y/half)),
              float(np.min((1-x)/half)), float(np.min((1-y)/half))]
    i, j = np.triu_indices(len(pose), 1)
    if len(i):
        dx, dy = x[j]-x[i], y[j]-y[i]
        delta = theta[i]-theta[j]
        radius = (1+np.abs(np.cos(delta))+np.abs(np.sin(delta)))/2
        angles = np.stack((theta[i], theta[i]+math.pi/2,
                           theta[j], theta[j]+math.pi/2), axis=1)
        separation = np.max(np.abs(dx[:, None]*np.cos(angles)+
                                   dy[:, None]*np.sin(angles))/radius[:, None], axis=1)
        limits.append(float(np.min(separation)))
    return max(0.0, min(limits)-clearance)


def _grid_pose(n):
    return packing_pose(grid(n))


def template_pose(n, kind, rng):
    """Create a repeatable research-inspired topology in the unit box."""
    np, _ = _libraries()
    pose = _grid_pose(n)
    width = math.isqrt(n-1)+1
    cells = np.arange(n)
    row, col = cells//width, cells%width
    if kind in ("diagonal-1", "diagonal-2"):
        band = 1 if kind.endswith("1") else 2
        offset = int(rng.integers(-(width//2), width//2+1))
        chosen = np.abs((row-col)-offset) < band
        signs = np.where(((row+col)&1)==0, 1.0, -1.0)
        pose[chosen, 2] = signs[chosen]*(math.pi/4+rng.normal(0, 0.035, np.sum(chosen)))
        normal = np.array([1.0, 1.0])/math.sqrt(2)
        pose[chosen, :2] += rng.normal(0, 0.045/width, (np.sum(chosen), 1))*normal
    elif kind == "central-block":
        distance = np.sum((pose[:, :2]-0.5)**2, axis=1)
        block = min(n, max(4, max(2, width//2)**2))
        chosen = np.argsort(distance)[:block]
        angle = rng.choice([-1.0, 1.0])*rng.uniform(0.35, math.pi/4)
        pivot = pose[chosen, :2].mean(axis=0)
        rot = np.array([[math.cos(angle), -math.sin(angle)],
                        [math.sin(angle), math.cos(angle)]])
        pose[chosen, :2] = (pose[chosen, :2]-pivot)@rot.T+pivot
        pose[chosen, 2] = angle
    elif kind == "staggered-rows":
        pose[:, 0] += ((row&1)*2-1)*rng.uniform(0.08, 0.24)/width
        pose[:, 2] = ((row&1)*2-1)*rng.uniform(0.08, 0.32)
    elif kind == "tilted-strips":
        angle = rng.uniform(0.08, 0.32)
        pose[:, 2] = np.where((row&1)==0, angle, -angle)
        pose[:, 0] += np.where((row&1)==0, 1, -1)*rng.uniform(0.04, 0.18)/width
    elif kind != "grid-jitter":
        raise ValueError("unknown structured template")
    pose[:, :2] += rng.normal(0, 0.018/width, (n, 2))
    pose[:, :2] = np.clip(pose[:, :2], 1e-5, 1-1e-5)
    pose[:, 2] = (pose[:, 2]+math.pi/2)%math.pi-math.pi/2
    return pose


def perturb_pose(pose, rng, strength=0.04):
    np, _ = _libraries()
    result=np.asarray(pose, dtype=float).copy()
    n=len(result)
    k=min(n, int(rng.integers(2, min(8, n)+1))) if n>1 else 1
    chosen=rng.choice(n, k, replace=False)
    pivot=result[chosen, :2].mean(axis=0)
    angle=float(rng.normal(0, strength*3))
    rot=np.array([[math.cos(angle), -math.sin(angle)],
                  [math.sin(angle), math.cos(angle)]])
    result[chosen, :2]=(result[chosen, :2]-pivot)@rot.T+pivot
    result[chosen, :2]+=rng.normal(0, strength, (k, 2))
    result[chosen, 2]+=angle+rng.normal(0, strength*2, k)
    result[:, :2]=np.clip(result[:, :2], 1e-5, 1-1e-5)
    result[:, 2]=(result[:, 2]+math.pi/2)%math.pi-math.pi/2
    return result


def _candidate(pose, clearance):
    size=maximal_scale(pose, clearance)
    if size <= 0:
        return None
    packing=[Square(float(x/size), float(y/size), float(t)) for x, y, t in pose]
    try:
        side=validate(packing, len(packing))
    except ValueError:
        return None
    return packing, side, size


def inflation_quench(pose, model, maxiter=120):
    np, minimize = _libraries()
    n=model.n
    initial=maximal_scale(pose, model.clearance)
    lower=max(1e-4, 0.45/(math.isqrt(n-1)+1))
    upper=1/math.sqrt(n)
    z=np.append(np.asarray(pose, dtype=float).ravel(), min(upper, max(lower, initial)))
    bounds=[(0, 1), (0, 1), (-math.pi/2, math.pi/2)]*n+[(lower, upper)]
    for weight in (100.0, 10000.0, 1000000.0):
        def energy(v):
            margin, jac=model.evaluate(v)
            negative=np.minimum(margin, 0)
            grad=2*weight*(jac.T@negative)
            grad[-1]-=1
            return float(-v[-1]+weight*(negative@negative)), grad
        result=minimize(energy, z, jac=True, method="L-BFGS-B", bounds=bounds,
                        options={"maxiter":maxiter, "ftol":1e-13, "gtol":1e-8})
        z=result.x
    pose=np.asarray(z[:-1]).reshape(n, 3)
    candidate=_candidate(pose, model.clearance*2)
    margin, _=model.evaluate(z)
    return candidate, pose, {"local_success":bool(result.success),
        "optimized_scale":float(z[-1]), "feasible_scale":candidate[2] if candidate else 0.0,
        "min_margin":float(np.min(margin))}


def solve_structured(n, config=None, initial=None, callback=None):
    """Run structured inflation trials and always return a valid incumbent."""
    np, _ = _libraries()
    config=config or StructuredConfig()
    if (not 1<=n<=1000 or config.trials<1 or config.maxiter<1 or
            not math.isfinite(config.seconds) or config.seconds<=0 or
            not math.isfinite(config.temperature) or config.temperature<0 or
            not 0<config.clearance<0.01):
        raise ValueError("invalid structured search configuration")
    baseline=grid(n)
    best=baseline
    best_side=validate(best, n)
    if initial is not None:
        supplied_side=validate(initial, n)
        if supplied_side<best_side:
            best, best_side=list(initial), supplied_side
    initial_side=best_side
    current_pose=packing_pose(best)
    current_side=best_side
    rng=np.random.default_rng(config.seed)
    started=time.perf_counter()
    model=InflationModel(n, config.clearance, started+config.seconds)
    kinds=("incumbent", "diagonal-1", "central-block", "staggered-rows",
           "diagonal-2", "tilted-strips", "grid-jitter")
    history=[]
    status="trial-limit"
    if abs(best_side-math.sqrt(n))<1e-12:
        status="area-bound"
    else:
        try:
            for trial in range(config.trials):
                kind=kinds[trial%len(kinds)]
                if kind=="incumbent":
                    strength=(0.008, 0.02, 0.05)[(trial//len(kinds))%3]
                    pose=perturb_pose(current_pose, rng, strength)
                else:
                    pose=template_pose(n, kind, rng)
                before=model.calls
                candidate, walk_pose, diagnostic=inflation_quench(pose, model, config.maxiter)
                candidate_side=candidate[1] if candidate else math.inf
                accepted=False
                if candidate is not None and metropolis(current_side, candidate_side,
                                                        config.temperature, float(rng.random())):
                    accepted=True
                    current_side=candidate_side
                    current_pose=packing_pose(candidate[0])
                elif candidate is None:
                    current_pose=walk_pose
                if candidate is not None and candidate_side<best_side-1e-10:
                    best, best_side=candidate[0], candidate_side
                row={"trial":trial, "template":kind, "accepted":accepted,
                     "candidate_L":candidate_side if candidate else None,
                     "best_L":best_side, "seconds":time.perf_counter()-started,
                     "trial_evaluations":model.calls-before, **diagnostic}
                history.append(row)
                if callback:
                    callback(best, row)
        except BudgetExpired:
            status="time-budget"
    validate(best, n)
    return best, {"algorithm":"structured-maximal-inflation", "n":n,
        "seed":config.seed, "initial_L":initial_side, "best_L":best_side,
        "improved":best_side<initial_side-1e-9, "seconds":time.perf_counter()-started,
        "status":status, "completed_trials":len(history), "evaluations":model.calls,
        "config":asdict(config), "history":history}
