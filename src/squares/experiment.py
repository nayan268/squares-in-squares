"""Reproducible solver benchmarks, with references kept separate from search."""
import csv
import hashlib
import json
import math
import platform
import statistics
from dataclasses import replace
from pathlib import Path
from .baseline import grid
from .constructions import five_square_reference
from .geometry import container_side, validate
from .official import check_submission
from .optimize import solve, SolverConfig
from .plot import plot_packings
from .submission import write_submission, score


def _write_json(path, value):
    with Path(path).open("x",encoding="utf-8") as handle:
        json.dump(value,handle,indent=2,allow_nan=False)


def benchmark(output, cases=(5,10,11), seeds=(0,1,2), config=None, metric=None):
    config=config or SolverConfig(restarts=6,maxiter=150,squeeze_steps=4)
    if not cases or not seeds or len(set(cases))!=len(cases) or len(set(seeds))!=len(seeds):
        raise ValueError("cases and seeds must be nonempty and unique")
    for n in cases:
        grid(n)  # Validate before creating the run directory.
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    import numpy,scipy
    versions={"python":platform.python_version(),"numpy":numpy.__version__,
              "scipy":scipy.__version__}
    source_dir=output/"source"
    source_dir.mkdir()
    hashes={}
    for name in ("optimize.py","geometry.py","experiment.py","official.py",
                 "baseline.py","constructions.py","submission.py","plot.py"):
        raw=Path(__file__).with_name(name).read_bytes()
        (source_dir/name).write_bytes(raw)
        hashes[name]=hashlib.sha256(raw).hexdigest()
    _write_json(output/"environment.json",{"versions":versions,"source_hashes":hashes,
                                         "cases":list(cases),"seeds":list(seeds),
                                         "reference_used_as_start":False})
    reference=five_square_reference()
    write_submission({5:reference},output/"reference-five.csv")
    if metric:
        _write_json(output/"reference-official.json",
                    check_submission(output/"reference-five.csv",metric))
    plot_packings([("Grid",grid(5)),("Known construction",reference)],
                  output/"reference-five.svg",title="First benchmark: five squares")
    plot_packings([("Grid",grid(5)),("Known construction",reference)],
                  output/"reference-five.png",title="First benchmark: five squares")
    results=[]
    best={n:grid(n) for n in cases}
    for n in cases:
        for seed in seeds:
            packing,result=solve(n,replace(config,seed=seed))
            validate(packing,n)
            stem=f"n{n}-seed{seed}"
            write_submission({n:packing},output/(stem+".csv"))
            if metric:
                result["official"]=check_submission(output/(stem+".csv"),metric)
                if not result["official"]["all_feasible"]:
                    result["rejected_by_official"]=True
            result["improved"]=(not result.get("rejected_by_official",False) and
                                result["best_L"] < result["initial_L"]-1e-7)
            result["n5_target_hit"]=(not result.get("rejected_by_official",False) and
                                     abs(result["best_L"]-(2+1/math.sqrt(2)))<1e-5
                                     if n==5 else None)
            _write_json(output/(stem+".json"),result)
            if (not result.get("rejected_by_official") and
                    result["best_L"] < container_side(best[n])):
                best[n]=packing
            results.append(result)
            print(f"n={n} seed={seed}: L={result['best_L']:.9f}; "
                  f"improved={result['improved']}; seconds={result['seconds']:.2f}",flush=True)
    write_submission(best,output/"best.csv")
    details=[]
    for n in cases:
        rows=[r for r in results if r["n"]==n]
        details.append({"n":n,"grid_L":container_side(grid(n)),
                        "best_L":container_side(best[n]),
                        "improved_runs":sum(r["improved"] for r in rows),
                        "runs":len(rows),
                        "median_seconds":statistics.median(r["seconds"] for r in rows),
                        "target_hits":sum(r["n5_target_hit"] is True for r in rows) if n==5 else None})
    with (output/"summary.csv").open("x",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    summary={"cases":details,"total_seconds":sum(r["seconds"] for r in results),
             "competition_score_estimate":score(best)["score"],
             "grid_score_estimate":score({})["score"],
             "reference_used_as_start":False}
    if metric:
        summary["official"]=check_submission(output/"best.csv",metric)
    _write_json(output/"summary.json",summary)
    panels=[]
    for n in cases:
        panels.extend([(f"Grid, n={n}",grid(n)),(f"Search result, n={n}",best[n])])
    plot_packings(panels,output/"comparison.svg",title="Grid baseline and blind search")
    plot_packings(panels,output/"comparison.png",title="Grid baseline and blind search")
    print(json.dumps(summary,indent=2),flush=True)
    return summary
