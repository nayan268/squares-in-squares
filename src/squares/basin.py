"""Basin hopping with collective proposals, analytic derivatives and a deadline.

The loop keeps a penalized exploration state separate from its valid incumbent.
SciPy provides L-BFGS-B and SLSQP local minimization; no record packing is loaded.
"""
import math
import time
from dataclasses import dataclass, asdict
from .baseline import grid
from .geometry import Square, validate
from .optimize import _libraries, _safe_candidate


class BudgetExpired(Exception):
    pass


@dataclass
class BasinConfig:
    seed: int = 0
    hops: int = 200
    seconds: float = 60.0
    maxiter: int = 160
    temperature: float = 0.04
    clearance: float = 1e-7
    max_expansion: float = 0.25
    soft_walk: bool = True


class PoseModel:
    """SAT/wall margins and analytic derivatives at differentiable points."""
    def __init__(self, n, clearance=1e-7, deadline=None):
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
        side = z[-1]
        c, s = np.cos(theta), np.sin(theta)
        half = (np.abs(c) + np.abs(s)) / 2
        dh = (-np.sign(c)*s + np.sign(s)*c) / 2
        walls = np.concatenate((x-half, y-half, side-x-half, side-y-half))
        jac = np.zeros((4*n+len(self.i), 3*n+1))
        r = np.arange(n)
        jac[r, 3*r] = 1
        jac[n+r, 3*r+1] = 1
        jac[2*n+r, 3*r] = -1
        jac[3*n+r, 3*r+1] = -1
        for k in range(4):
            jac[k*n+r, 3*r+2] = -dh
        jac[2*n:4*n, -1] = 1
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
        gaps = np.abs(dot)-radius[:, None]
        k = np.argmax(gaps, axis=1)
        q = np.arange(len(i))
        co, si = ca[q,k], sa[q,k]
        sign = np.sign(dot[q,k])
        rotation = sign*(-dx*si+dy*co)
        rr = 4*n+q
        jac[rr,3*i] = -sign*co
        jac[rr,3*i+1] = -sign*si
        jac[rr,3*j] = sign*co
        jac[rr,3*j+1] = sign*si
        jac[rr,3*i+2] = -dr+np.where(k<2,rotation,0)
        jac[rr,3*j+2] = dr+np.where(k>=2,rotation,0)
        return np.concatenate((walls,gaps[q,k]))-self.clearance, jac


def metropolis(old, new, temperature, uniform):
    """Metropolis test on the selected walk energy."""
    return new <= old or (temperature > 0 and
                          uniform < math.exp(-(new-old)/temperature))


def propose(packing, rng, hop, upper):
    np, _ = _libraries()
    a = np.array([[s.x,s.y,s.angle] for s in packing])
    n = len(a)
    mode = hop % 6
    scale = (0.10,0.25,0.45)[(hop//6)%3]
    k = min(n, int(rng.integers(2, min(6,n)+1))) if n>1 else 1
    ids = rng.choice(n,k,replace=False)
    if mode == 0:
        pivot = a[ids,:2].mean(axis=0)
        angle = rng.uniform(-0.6,0.6)
        rot = np.array([[math.cos(angle),-math.sin(angle)],
                        [math.sin(angle),math.cos(angle)]])
        a[ids,:2] = (a[ids,:2]-pivot) @ rot.T+pivot+rng.normal(0,scale,2)
        a[ids,2] += angle+rng.normal(0,0.08,k)
        label = "collective-rotation"
    elif mode == 1:
        a[ids,:2] += rng.normal(0,scale,(k,2))
        a[ids,2] += rng.choice([-1,1],k)*rng.uniform(0.25,0.8,k)
        label = "tilt-group"
    elif mode == 2:
        width = math.isqrt(n-1)+1
        cells = rng.choice(width*width,n,replace=False)
        a = np.column_stack((cells%width+0.5,cells//width+0.5,
                             rng.uniform(-math.pi/4,math.pi/4,n)))
        a[:,:2] += rng.normal(0,0.12,(n,2))
        label = "random-lattice-restart"
    elif mode == 3:
        a[ids,:2] = rng.uniform(0.5,upper-0.5,(k,2))
        a[ids,2] = rng.uniform(-math.pi/2,math.pi/2,k)
        label = "reinsert-group"
    elif mode == 4:
        a[:,:2] *= rng.uniform(1.01,1.06)
        a[:,:2] += rng.normal(0,scale/2,(n,2))
        a[:,2] += rng.normal(0,0.25,n)
        label = "expand-and-shake"
    else:
        a[:,:2] = rng.uniform(0.5,upper-0.5,(n,2))
        a[:,2] = rng.uniform(-math.pi/2,math.pi/2,n)
        label = "random-pose-restart"
    a[:,:2] = np.clip(a[:,:2],0,upper)
    a[:,2] = (a[:,2]+math.pi/2)%math.pi-math.pi/2
    return a, label


def quench(pose, model, upper, maxiter=160):
    np, minimize = _libraries()
    n = model.n
    extent = (np.abs(np.cos(pose[:,2]))+np.abs(np.sin(pose[:,2])))/2
    side = min(upper,max(math.sqrt(n),float(np.max(pose[:,:2]+extent[:,None]))))
    z = np.append(pose.ravel(),side)
    bounds = [(0,upper),(0,upper),(-math.pi,math.pi)]*n+[(math.sqrt(n),upper)]
    for weight in (40.0,400.0,4000.0):
        def energy(v):
            margin,jac = model.evaluate(v)
            negative = np.minimum(margin,0)
            grad = 2*weight*(jac.T @ negative)
            grad[-1] += 1
            return float(v[-1]+weight*(negative @ negative)),grad
        result = minimize(energy,z,jac=True,method="L-BFGS-B",bounds=bounds,
                          options={"maxiter":min(100,maxiter),"ftol":1e-12,"gtol":1e-7})
        z = result.x
    unit = np.zeros(3*n+1)
    unit[-1] = 1
    # Cache repeated constraint/jacobian evaluations at identical coordinates.
    cached_z, cached = None, None
    def both(v):
        nonlocal cached_z,cached
        if cached_z is None or not np.array_equal(v,cached_z):
            cached_z = v.copy()
            cached = model.evaluate(v)
        return cached
    result = minimize(lambda v: float(v[-1]),z,jac=lambda v:unit,
                      method="SLSQP",bounds=bounds,
                      constraints={"type":"ineq","fun":lambda v:both(v)[0],
                                   "jac":lambda v:both(v)[1]},
                      options={"maxiter":maxiter,"ftol":1e-10})
    margin,_ = model.evaluate(result.x)
    candidate = _safe_candidate(result.x[:-1],n,model.clearance)
    violation = np.minimum(margin,0)
    diagnostic = {"local_success":bool(result.success),"min_margin":float(np.min(margin)),
                  "walk_vector":result.x[:-1].copy(),
                  "walk_energy":float(result.x[-1]+4000*(violation @ violation))}
    if candidate is None or float(np.min(margin)) < -1e-8:
        return None, diagnostic
    diagnostic["walk_energy"] = candidate[1]
    return candidate, diagnostic


def solve_basin(n=11, config=None, initial=None, callback=None):
    np, _ = _libraries()
    config = config or BasinConfig()
    baseline = grid(n)
    if (config.hops<1 or config.maxiter<1 or not math.isfinite(config.seconds)
            or config.seconds<=0 or not math.isfinite(config.temperature)
            or config.temperature<0 or not 0<config.clearance<0.01
            or not 0<=config.max_expansion<=2):
        raise ValueError("invalid basin configuration")
    best = baseline
    best_side = validate(best,n)
    current = list(initial) if initial is not None else baseline
    current_side = validate(current,n)
    if current_side<best_side:
        best,best_side = current,current_side
    current_energy = current_side
    initial_side = best_side
    started = time.perf_counter()
    model = PoseModel(n,config.clearance,started+config.seconds)
    rng = np.random.default_rng(config.seed)
    upper = max(validate(baseline,n),current_side)+config.max_expansion
    history = []
    status = "hop-limit"
    if abs(best_side-math.sqrt(n))<1e-12:
        status = "area-bound"
    else:
        try:
            for hop in range(config.hops):
                pose,move = propose(current,rng,hop,upper)
                candidate,diagnostic = quench(pose,model,upper,config.maxiter)
                walk_vector = diagnostic.pop("walk_vector")
                energy = diagnostic["walk_energy"]
                delta = energy-current_energy
                accepted = False
                if candidate is not None or config.soft_walk:
                    accepted = metropolis(current_energy,energy,config.temperature,float(rng.random()))
                if accepted:
                    current_energy = energy
                    if candidate is not None:
                        current,current_side = candidate
                    else:
                        current = [Square(float(x),float(y),float(t))
                                   for x,y,t in walk_vector.reshape(n,3)]
                        current_side = None
                if candidate is not None and candidate[1]<best_side:
                    best,best_side = candidate
                row = {"hop":hop,"move":move,"valid":candidate is not None,
                       "candidate_L":candidate[1] if candidate else None,
                       "accepted":accepted,"accepted_uphill":bool(accepted and delta>1e-8),
                       "current_L":current_side,"current_energy":current_energy,"best_L":best_side,
                       "seconds":time.perf_counter()-started,**diagnostic}
                history.append(row)
                if callback:
                    callback(best,row)
        except BudgetExpired:
            status = "time-budget"
    validate(best,n)
    return best, {"algorithm":"basin-hopping-analytic","n":n,"initial_L":initial_side,
                  "best_L":best_side,"seconds":time.perf_counter()-started,
                  "config":asdict(config),"status":status,"completed_hops":len(history),
                  "accepted_hops":sum(r["accepted"] for r in history),
                  "accepted_uphill":sum(r["accepted_uphill"] for r in history),
                  "margin_evaluations":model.calls,"reference_used_as_start":initial is not None,
                  "history":history}
