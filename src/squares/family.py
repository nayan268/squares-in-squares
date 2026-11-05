"""Low-dimensional row-family searches for cases just above a square.

Every candidate is a set of unit squares on a common rotated lattice.  Rows
remain one square apart in lattice coordinates, so changing row shifts and the
global rotation cannot introduce overlap.  This lets Powell search a handful
of pattern parameters instead of moving all 3*n square variables.
"""
import math
import time
from dataclasses import asdict, dataclass

from .baseline import grid
from .basin import BudgetExpired
from .geometry import Square, validate
from .optimize import _libraries
from .structured import _candidate, maximal_scale


@dataclass
class FamilyConfig:
    seed: int = 0
    seconds: float = 20.0
    trials: int = 1000
    maxiter: int = 180
    clearance: float = 1e-8


def balanced_row_counts(n, variant=0, rng=None):
    """Return a seeded ordering of the most balanced row counts for n."""
    np, _ = _libraries()
    if not 1 <= n <= 1000:
        raise ValueError("n must be from 1 through 1000")
    rows = math.isqrt(n - 1) + 1
    short, extra = divmod(n, rows)
    counts = [short] * rows
    middle = (rows - 1) / 2
    mode = variant % 4
    if mode == 0:
        order = sorted(range(rows), key=lambda i: (abs(i - middle), i))
    elif mode == 1:
        order = sorted(range(rows), key=lambda i: (-abs(i - middle), i))
    elif mode == 2:
        order = []
        for i in range((rows + 1) // 2):
            order.append(i)
            if rows - 1 - i != i:
                order.append(rows - 1 - i)
    else:
        generator = rng if rng is not None else np.random.default_rng(variant)
        order = list(map(int, generator.permutation(rows)))
    for row in order[:extra]:
        counts[row] += 1
    return tuple(counts)


def _centers(counts, shifts):
    np, _ = _libraries()
    rows = len(counts)
    centers = []
    for row, count in enumerate(counts):
        y = row - (rows - 1) / 2
        centers.extend((column - (count - 1) / 2 + shifts[row], y)
                       for column in range(count))
    return np.asarray(centers, dtype=float)


def family_side(counts, parameters):
    """Exact axis-aligned container side for one common-angle row family."""
    np, _ = _libraries()
    parameters = np.asarray(parameters, dtype=float)
    if parameters.shape != (len(counts) + 1,):
        raise ValueError("incorrect family parameter shape")
    theta = float(parameters[0])
    centers = _centers(counts, parameters[1:])
    c, s = math.cos(theta), math.sin(theta)
    x = centers[:, 0] * c - centers[:, 1] * s
    y = centers[:, 0] * s + centers[:, 1] * c
    extent = abs(c) + abs(s)
    return max(float(np.ptp(x)) + extent, float(np.ptp(y)) + extent)


def family_packing(counts, parameters, clearance=1e-8):
    """Build a valid unit-square packing with small numerical clearance."""
    np, _ = _libraries()
    theta = float(parameters[0])
    centers = _centers(counts, parameters[1:])
    c, s = math.cos(theta), math.sin(theta)
    rotated = np.column_stack((centers[:, 0] * c - centers[:, 1] * s,
                               centers[:, 0] * s + centers[:, 1] * c))
    rotated *= 1 + clearance
    half = (abs(c) + abs(s)) / 2
    rotated[:, 0] -= float(np.min(rotated[:, 0])) - half
    rotated[:, 1] -= float(np.min(rotated[:, 1])) - half
    packing = [Square(float(x), float(y), theta) for x, y in rotated]
    validate(packing, sum(counts))
    return packing


def _initial_parameters(counts, mode, rng):
    np, _ = _libraries()
    rows = len(counts)
    missing = np.asarray([max(counts) - count for count in counts], dtype=float)
    shifts = np.zeros(rows)
    if mode % 4 == 1:
        shifts = missing / 2
    elif mode % 4 == 2:
        shifts = ((np.arange(rows) & 1) * 2 - 1) * missing / 2
    elif mode % 4 == 3:
        shifts = rng.uniform(-0.55, 0.55, rows)
    shifts += rng.normal(0, 0.04, rows)
    theta = rng.uniform(-0.20, 0.20)
    return np.concatenate(([theta], shifts))


def interlocked_pose(counts, parameters):
    """Unit-container pose with global tilt and alternating row angles."""
    np, _ = _libraries()
    parameters = np.asarray(parameters, dtype=float)
    if parameters.shape != (len(counts) + 4,):
        raise ValueError("incorrect interlocked parameter shape")
    tilt, angle, x_step, y_step = parameters[:4]
    offsets = parameters[4:]
    centers = []
    angles = []
    middle = (len(counts) - 1) / 2
    for row, count in enumerate(counts):
        y = (row - middle) * y_step
        sign = 1 if row % 2 == 0 else -1
        for column in range(count):
            x = (column - (count - 1) / 2 + offsets[row]) * x_step
            centers.append((x, y))
            angles.append(tilt + sign * angle)
    centers = np.asarray(centers, dtype=float)
    c, s = math.cos(tilt), math.sin(tilt)
    rotated = np.column_stack((centers[:, 0] * c - centers[:, 1] * s,
                               centers[:, 0] * s + centers[:, 1] * c))
    rotated += 0.5
    return np.column_stack((rotated, np.asarray(angles)))


def _interlocked_initial(counts, mode, rng):
    np, _ = _libraries()
    rows = len(counts)
    width = max(counts)
    missing = np.asarray([width - count for count in counts], dtype=float)
    offsets = np.zeros(rows)
    if mode % 4 == 1:
        offsets = missing / 2
    elif mode % 4 == 2:
        offsets = ((np.arange(rows) & 1) * 2 - 1) * (0.28 + missing / 2)
    elif mode % 4 == 3:
        offsets = rng.uniform(-0.65, 0.65, rows)
    offsets += rng.normal(0, 0.05, rows)
    scale = 1 / rows
    return np.concatenate(([rng.uniform(-0.08, 0.08),
                            rng.uniform(0.03, 0.28),
                            scale * rng.uniform(0.88, 1.08),
                            scale * rng.uniform(0.88, 1.08)], offsets))


def defect_row_counts(n, variant=0, rng=None):
    """Return a row topology with one or more displaced-square defects."""
    np, _ = _libraries()
    counts = list(balanced_row_counts(n, variant, rng))
    rows = len(counts)
    if rows < 2:
        return tuple(counts)
    generator = rng if rng is not None else np.random.default_rng(variant)
    order = list(map(int, generator.permutation(rows)))
    pairs = min(1 + (variant // 4) % 2, rows // 2)
    for pair in range(pairs):
        donor = order[2 * pair]
        receiver = order[2 * pair + 1]
        if counts[donor] > 1:
            counts[donor] -= 1
            counts[receiver] += 1
    return tuple(counts)


def defect_pose(counts, parameters):
    """Pose for mixed rows with independent shifts, gaps, and row angles."""
    np, _ = _libraries()
    rows = len(counts)
    parameters = np.asarray(parameters, dtype=float)
    if parameters.shape != (3 * rows + 1,):
        raise ValueError("incorrect defect parameter shape")
    tilt, x_step = parameters[:2]
    offsets = parameters[2:2 + rows]
    gaps = parameters[2 + rows:1 + 2 * rows]
    row_angles = parameters[1 + 2 * rows:]
    y_positions = np.concatenate(([0.0], np.cumsum(gaps)))
    y_positions -= float(np.mean(y_positions))
    centers = []
    angles = []
    for row, count in enumerate(counts):
        for column in range(count):
            centers.append(((column - (count - 1) / 2 + offsets[row]) * x_step,
                            y_positions[row]))
            angles.append(tilt + row_angles[row])
    centers = np.asarray(centers, dtype=float)
    c, s = math.cos(tilt), math.sin(tilt)
    centers = np.column_stack((centers[:, 0] * c - centers[:, 1] * s,
                               centers[:, 0] * s + centers[:, 1] * c))
    centers += 0.5
    return np.column_stack((centers, np.asarray(angles)))


def _defect_initial(counts, mode, rng):
    np, _ = _libraries()
    rows = len(counts)
    width = max(counts)
    missing = np.asarray([width - count for count in counts], dtype=float)
    offsets = (missing / 2 if mode % 3 == 0 else
               ((np.arange(rows) & 1) * 2 - 1) * (0.20 + missing / 2)
               if mode % 3 == 1 else rng.uniform(-0.8, 0.8, rows))
    offsets += rng.normal(0, 0.04, rows)
    offsets = np.clip(offsets, -1.24, 1.24)
    step = 1 / rows
    gaps = step * rng.uniform(0.90, 1.10, rows - 1)
    amplitude = rng.uniform(0.05, 0.30)
    row_angles = ((np.arange(rows) & 1) * 2 - 1) * amplitude
    row_angles += rng.normal(0, 0.025, rows)
    row_angles = np.clip(row_angles, -0.54, 0.54)
    return np.concatenate(([rng.uniform(-0.08, 0.08),
                            step * rng.uniform(0.90, 1.10)],
                           offsets, gaps, row_angles))


def solve_defect_family(n, config=None, initial=None, callback=None):
    """Search mixed-row defect topologies and retain a valid incumbent."""
    np, minimize = _libraries()
    config = config or FamilyConfig()
    if (not 1 <= n <= 1000 or config.trials < 1 or config.maxiter < 1 or
            not math.isfinite(config.seconds) or config.seconds <= 0 or
            not 0 < config.clearance < 0.01):
        raise ValueError("invalid defect-family search configuration")
    best = grid(n)
    best_side = validate(best, n)
    if initial is not None:
        supplied = validate(initial, n)
        if supplied < best_side:
            best, best_side = list(initial), supplied
    initial_side = best_side
    rng = np.random.default_rng(config.seed)
    started = time.perf_counter()
    deadline = started + config.seconds
    history = []
    evaluations = 0
    status = "trial-limit"
    try:
        for trial in range(config.trials):
            if time.perf_counter() >= deadline:
                raise BudgetExpired()
            counts = defect_row_counts(n, trial, rng)
            rows = len(counts)
            x0 = _defect_initial(counts, trial, rng)
            trial_best = [0.0, x0.copy()]

            def objective(parameters):
                nonlocal evaluations
                if time.perf_counter() >= deadline:
                    raise BudgetExpired()
                evaluations += 1
                scale = maximal_scale(defect_pose(counts, parameters),
                                      config.clearance)
                if scale > trial_best[0]:
                    trial_best[:] = [scale, np.asarray(parameters).copy()]
                return -scale

            expired = False
            bounds = ([(-0.30, 0.30), (0.60 / rows, 1.35 / rows)] +
                      [(-1.25, 1.25)] * rows +
                      [(0.60 / rows, 1.35 / rows)] * (rows - 1) +
                      [(-0.55, 0.55)] * rows)
            try:
                result = minimize(objective, x0, method="Powell", bounds=bounds,
                    options={"maxiter": config.maxiter,
                             "maxfev": config.maxiter * 12,
                             "xtol": 1e-5, "ftol": 1e-8})
                parameters = (result.x if -result.fun >= trial_best[0]
                              else trial_best[1])
                local_success = bool(result.success)
            except BudgetExpired:
                parameters = trial_best[1]
                local_success = False
                expired = True
            candidate_data = _candidate(defect_pose(counts, parameters),
                                        config.clearance * 2)
            candidate = candidate_data[0] if candidate_data else None
            candidate_side = candidate_data[1] if candidate_data else math.inf
            if candidate_side < best_side - 1e-10:
                best, best_side = candidate, candidate_side
            row = {"trial": trial, "counts": list(counts),
                   "candidate_L": candidate_side if candidate is not None else None,
                   "best_L": best_side,
                   "seconds": time.perf_counter() - started,
                   "local_success": local_success, "evaluations": evaluations}
            history.append(row)
            if callback:
                callback(best, row)
            if expired:
                raise BudgetExpired()
    except BudgetExpired:
        status = "time-budget"
    validate(best, n)
    return best, {"algorithm": "mixed-row-defect-family", "n": n,
        "seed": config.seed, "initial_L": initial_side, "best_L": best_side,
        "improved": best_side < initial_side - 1e-9,
        "seconds": time.perf_counter() - started, "status": status,
        "completed_trials": len(history), "evaluations": evaluations,
        "config": asdict(config), "history": history}


def solve_family(n, config=None, initial=None, callback=None):
    """Search balanced row families and always retain a valid incumbent."""
    np, minimize = _libraries()
    config = config or FamilyConfig()
    if (not 1 <= n <= 1000 or config.trials < 1 or config.maxiter < 1 or
            not math.isfinite(config.seconds) or config.seconds <= 0 or
            not 0 < config.clearance < 0.01):
        raise ValueError("invalid family search configuration")
    best = grid(n)
    best_side = validate(best, n)
    if initial is not None:
        supplied = validate(initial, n)
        if supplied < best_side:
            best, best_side = list(initial), supplied
    initial_side = best_side
    rng = np.random.default_rng(config.seed)
    started = time.perf_counter()
    deadline = started + config.seconds
    history = []
    evaluations = 0
    status = "trial-limit"
    try:
        for trial in range(config.trials):
            if time.perf_counter() >= deadline:
                raise BudgetExpired()
            counts = balanced_row_counts(n, trial, rng)
            x0 = _interlocked_initial(counts, trial, rng)
            trial_best = [0.0, x0.copy()]

            def objective(parameters):
                nonlocal evaluations
                if time.perf_counter() >= deadline:
                    raise BudgetExpired()
                evaluations += 1
                scale = maximal_scale(interlocked_pose(counts, parameters),
                                      config.clearance)
                if scale > trial_best[0]:
                    trial_best[:] = [scale, np.asarray(parameters).copy()]
                return -scale

            expired = False
            try:
                result = minimize(objective, x0, method="Powell",
                    bounds=[(-0.30, 0.30), (0.0, math.pi / 4),
                            (0.60 / len(counts), 1.25 / len(counts)),
                            (0.60 / len(counts), 1.25 / len(counts))] +
                           [(-0.9, 0.9)] * len(counts),
                    options={"maxiter": config.maxiter,
                             "maxfev": config.maxiter * 10,
                             "xtol": 1e-5, "ftol": 1e-8})
                parameters = result.x if -result.fun >= trial_best[0] else trial_best[1]
                local_success = bool(result.success)
            except BudgetExpired:
                parameters = trial_best[1]
                local_success = False
                expired = True
            candidate_data = _candidate(interlocked_pose(counts, parameters),
                                        config.clearance * 2)
            candidate = candidate_data[0] if candidate_data else None
            candidate_side = candidate_data[1] if candidate_data else math.inf
            if candidate_side < best_side - 1e-10:
                best, best_side = candidate, candidate_side
            row = {"trial": trial, "counts": list(counts),
                   "candidate_L": candidate_side if candidate is not None else None,
                   "best_L": best_side,
                   "seconds": time.perf_counter() - started,
                   "local_success": local_success,
                   "evaluations": evaluations}
            history.append(row)
            if callback:
                callback(best, row)
            if expired:
                raise BudgetExpired()
    except BudgetExpired:
        status = "time-budget"
    validate(best, n)
    return best, {"algorithm": "interlocked-row-family", "n": n,
        "seed": config.seed, "initial_L": initial_side, "best_L": best_side,
        "improved": best_side < initial_side - 1e-9,
        "seconds": time.perf_counter() - started, "status": status,
        "completed_trials": len(history), "evaluations": evaluations,
        "config": asdict(config), "history": history}
