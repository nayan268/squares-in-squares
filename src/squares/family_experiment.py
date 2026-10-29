"""Resumable selected-case benchmark for balanced row families."""
import hashlib
import json
from pathlib import Path

from .family import FamilyConfig, solve_family
from .geometry import Square, validate
from .official import load_metric
from .research_seeds import propagated_seed
from .structured_experiment import _atomic, _certify
from .submission import read_submission, write_submission


def benchmark(output, cases, seeds, seconds, initial_path, metric_path=None,
              maxiter=180, resume=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cases = tuple(dict.fromkeys(map(int, cases)))
    seeds = tuple(dict.fromkeys(map(int, seeds)))
    if not cases or any(not 1 <= n <= 1000 for n in cases) or not seeds or seconds <= 0:
        raise ValueError("invalid cases, seeds, or time budget")
    initial_path = Path(initial_path)
    metric_path = Path(metric_path) if metric_path else None
    config = {"algorithm": "interlocked-row-family", "cases": cases, "seeds": seeds,
              "seconds": seconds, "maxiter": maxiter, "initial": str(initial_path),
              "initial_sha256": hashlib.sha256(initial_path.read_bytes()).hexdigest(),
              "metric": str(metric_path) if metric_path else None,
              "metric_sha256": hashlib.sha256(metric_path.read_bytes()).hexdigest()
                               if metric_path else None}
    settings = output / "settings.json"
    comparable = json.loads(json.dumps(config))
    if settings.exists():
        if not resume or json.loads(settings.read_text()) != comparable:
            raise ValueError("existing experiment differs; use matching settings with --resume")
    else:
        if resume:
            raise ValueError("no experiment exists to resume")
        _atomic(settings, config)
    initial = read_submission(initial_path)
    if any(n not in initial for n in cases):
        raise ValueError("initial submission is missing a requested case")
    metric = load_metric(metric_path)[0] if metric_path else None
    run_dir = output / "runs"
    run_dir.mkdir(exist_ok=True)
    starts = {n: list(initial[n]) for n in cases}
    for n in cases:
        try:
            propagated = propagated_seed(n)
        except KeyError:
            propagated = None
        if propagated is not None and validate(propagated,n) < validate(starts[n],n):
            starts[n] = propagated
    if metric is not None:
        for n in cases:
            if not _certify(metric,n,starts[n])["feasible"]:
                starts[n] = list(initial[n])
    best = {n: list(starts[n]) for n in cases}
    records = []
    for n in cases:
        for seed in seeds:
            checkpoint = run_dir / "n{:04d}-seed{:06d}.json".format(n, seed)
            if checkpoint.exists():
                record = json.loads(checkpoint.read_text())
                packing = [Square(*row) for row in record["packing"]]
            else:
                packing, report = solve_family(n, FamilyConfig(seed=seed,
                    seconds=seconds, maxiter=maxiter), initial=starts[n])
                certification = _certify(metric, n, packing)
                if not certification["feasible"]:
                    packing = list(initial[n])
                    report["rejected_by_official_metric"] = certification
                    certification = _certify(metric, n, packing)
                record = {"n": n, "seed": seed, "L": validate(packing, n),
                          "packing": [[s.x, s.y, s.angle] for s in packing],
                          "certification": certification, "report": report}
                _atomic(checkpoint, record)
            records.append({k: v for k, v in record.items() if k != "packing"})
            if validate(packing, n) < validate(best[n], n):
                best[n] = packing
            _atomic(output / "progress.json", {"status": "running",
                "completed": len(records), "total": len(cases) * len(seeds),
                "last_n": n, "last_seed": seed})
    csv_path = output / "best-cases.csv"
    if csv_path.exists():
        csv_path.unlink()
    write_submission(best, csv_path)
    summary = {"status": "complete", "cases": list(cases), "seeds": list(seeds),
               "seconds_per_run": seconds, "runs": records,
               "best": {str(n): {"original_L": validate(initial[n], n),
                                  "family_seed_L": validate(starts[n], n),
                                  "best_L": validate(best[n], n),
                                  "family_seed_improved": validate(starts[n], n) <
                                      validate(initial[n], n) - 1e-9,
                                  "optimizer_improved": validate(best[n], n) <
                                      validate(starts[n], n) - 1e-9}
                        for n in cases}, "submission": str(csv_path)}
    _atomic(output / "summary.json", summary)
    _atomic(output / "progress.json", {"status": "complete",
        "completed": len(records), "total": len(records), "submission": str(csv_path)})
    return summary
