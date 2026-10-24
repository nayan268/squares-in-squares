"""Equal wall-clock comparisons against the original finite-difference solver."""
import csv
import hashlib
import json
import platform
import statistics
from pathlib import Path
from .baseline import grid
from .basin import solve_basin,BasinConfig
from .geometry import validate
from .official import check_submission
from .optimize import solve,SolverConfig
from .plot import plot_packings
from .submission import read_submission,write_submission


def compare(output,n=11,seeds=(20,21,22),seconds=60.0,metric=None,resume=False):
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    if seconds<=0:
        raise ValueError("seconds must be positive")
    grid(n)
    output=Path(output)
    if resume and not (output/"environment.json").is_file():
        raise ValueError("resume requires an existing benchmark checkpoint")
    output.mkdir(parents=True,exist_ok=resume)
    source=output/"source"
    source.mkdir(exist_ok=resume)
    hashes={}
    for name in ("basin.py","basin_experiment.py","optimize.py","geometry.py",
                 "baseline.py","submission.py","official.py","plot.py"):
        raw=Path(__file__).with_name(name).read_bytes()
        hashes[name]=hashlib.sha256(raw).hexdigest()
        if not resume:
            (source/name).write_bytes(raw)
    import numpy,scipy
    metric_hash=hashlib.sha256(Path(metric).read_bytes()).hexdigest() if metric else None
    def save(name,obj):
        with (output/name).open("x",encoding="utf-8") as f:
            json.dump(obj,f,indent=2,allow_nan=False)
    if resume:
        previous=json.loads((output/"environment.json").read_text())
        if (previous["n"]!=n or previous["seeds"]!=list(seeds)
                or previous["seconds_per_run"]!=seconds):
            raise ValueError("resume settings differ from the saved benchmark")
        previous_metric=previous.get("metric_sha256")
        if "metric_sha256" not in previous:
            prior_runs=sorted(output.glob("*-seed*.json"))
            if prior_runs:
                sample=json.loads(prior_runs[0].read_text())
                previous_metric=sample.get("official",{}).get("provenance",{}).get("source_sha256")
        if previous_metric!=metric_hash:
            raise ValueError("metric source differs from the saved benchmark")
        for name,digest in hashes.items():
            if name!="basin_experiment.py" and previous["source_hashes"][name]!=digest:
                raise ValueError("solver source changed since checkpoint: "+name)
        (output/"resume-runner.py").write_bytes(Path(__file__).read_bytes())
        (output/"resume-provenance.json").write_text(json.dumps({
            "runner_sha256":hashes["basin_experiment.py"],
            "algorithm_hashes_verified":True},indent=2))
        if (output/"summary.json").exists():
            return json.loads((output/"summary.json").read_text())
    else:
        save("environment.json",{"python":platform.python_version(),"numpy":numpy.__version__,
                                "scipy":scipy.__version__,"source_hashes":hashes,
                                "n":n,"seeds":list(seeds),"seconds_per_run":seconds,
                                "metric_sha256":metric_hash,
                                "pilot_seeds":[0,1],"reference_used_as_start":False,
                                "comparison_scope":"new hopping plus analytic derivatives versus original finite-difference multistart"})
    rows=[]
    packings={"legacy":grid(n),"basin":grid(n)}
    for index,seed in enumerate(seeds):
        order=("legacy","basin") if index%2==0 else ("basin","legacy")
        for algorithm in order:
            stem=f"{algorithm}-seed{seed}"
            saved=resume and (output/(stem+".json")).exists()
            if saved:
                result=json.loads((output/(stem+".json")).read_text())
                p=read_submission(output/(stem+".csv"))[n]
                print("Resuming: reused "+stem,flush=True)
            else:
                if algorithm=="legacy":
                    p,result=solve(n,SolverConfig(seed=seed,restarts=100,maxiter=150,
                                                 squeeze_steps=3),time_limit=seconds)
                    result["algorithm"]="legacy"
                else:
                    last=validate(grid(n),n)
                    def checkpoint(p,row):
                        nonlocal last
                        if row["best_L"]<last-1e-8:
                            last=row["best_L"]
                            print(f"seed={seed} basin hop={row['hop']} new best L={last:.9f}",flush=True)
                    p,result=solve_basin(n,BasinConfig(seed=seed,seconds=seconds,hops=1000,
                                                     maxiter=140),callback=checkpoint)
            side=validate(p,n)
            stem=f"{algorithm}-seed{seed}"
            if not saved:
                write_submission({n:p},output/(stem+".csv"))
            if metric and not saved:
                result["official"]=check_submission(output/(stem+".csv"),metric)
            certified=metric is not None and result["official"]["all_feasible"]
            result["improved"]=side<validate(grid(n),n)-1e-7 and (metric is None or certified)
            if not saved:
                save(stem+".json",result)
            if (metric is None or certified) and side<validate(packings[algorithm],n):
                packings[algorithm]=p
            row={"algorithm":algorithm,"seed":seed,"L":side,"seconds":result["seconds"],
                 "improved":result["improved"],"official_feasible":certified if metric else None,
                 "completed_hops":result.get("completed_hops"),
                 "accepted_uphill":result.get("accepted_uphill")}
            rows.append(row)
            (output/"progress.json").write_text(json.dumps({"runs":rows},indent=2))
            print(json.dumps(row),flush=True)
    write_submission({n:packings["basin"]},output/"best-basin.csv")
    write_submission({n:packings["legacy"]},output/"best-legacy.csv")
    summary=[]
    for algorithm in ("legacy","basin"):
        selected=[r for r in rows if r["algorithm"]==algorithm]
        summary.append({"algorithm":algorithm,"best_L":min(r["L"] for r in selected),
                        "median_L":statistics.median(r["L"] for r in selected),
                        "improved_runs":sum(r["improved"] for r in selected),
                        "runs":len(selected),
                        "median_seconds":statistics.median(r["seconds"] for r in selected)})
    save("summary.json",{"n":n,"seconds_per_run":seconds,"seeds":list(seeds),
                         "summary":summary,"runs":rows})
    with (output/"summary.csv").open("x",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    panels=[("Grid",grid(n)),("Legacy",packings["legacy"]),("Basin hopping",packings["basin"])]
    plot_packings(panels,output/"comparison.svg",title=f"Basin hopping comparison: {n} squares")
    plot_packings(panels,output/"comparison.png",title=f"Basin hopping comparison: {n} squares")
    return {"summary":summary,"runs":rows}
