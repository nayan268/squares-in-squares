"""Standalone, resumable all-puzzle CPU job. No network or LLM calls."""
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path
from .baseline import grid
from .geometry import Square,validate
from .submission import read_submission,write_submission,score
from .warmstarts import extended_starts

_METRIC=None

def _init_worker(path):
    global _METRIC
    if path:
        from .official import load_metric
        _METRIC,_=load_metric(path)

def _certify(n,packing):
    if _METRIC is None:
        return {"kind":"local-only","feasible":True}
    rows=[{"square_id":str(i),"x":repr(float(s.x)),"y":repr(float(s.y)),
           "angle":repr(float(s.angle))} for i,s in enumerate(packing)]
    squares,side=_METRIC._build_squares(n,rows)
    feasible,reason=_METRIC.verify_puzzle(squares)
    return {"kind":"official-notebook","feasible":bool(feasible),"reason":reason,
            "L":_METRIC.mpmath.nstr(side,55) if feasible else None}

def _worker(task):
    n,initial,settings=task
    before=time.perf_counter()
    baseline=grid(n)
    baseline_side=validate(baseline,n)
    incumbent=initial or baseline
    if validate(incumbent,n)>baseline_side:
        incumbent=baseline
    if abs(baseline_side-math.sqrt(n))<1e-12:
        candidate=baseline
        report={"algorithm":"area-bound-grid","seconds":0,"completed_hops":0}
    elif n<=settings["dense_max_n"]:
        from .basin import solve_basin,BasinConfig
        candidate,report=solve_basin(n,BasinConfig(seed=settings["seed"]+n,
            seconds=settings["small_seconds"],hops=100000,maxiter=140),initial=incumbent)
    else:
        from .scalable import solve_scalable
        candidate,report=solve_scalable(n,seconds=settings["large_seconds"],
            seed=settings["seed"]+n,initial=incumbent,
            maxiter=settings["large_maxiter"],hops=100000)
    ordered=sorted((candidate,incumbent,baseline),key=lambda p:validate(p,n))
    selected=baseline
    rejected=[]
    certification=None
    for packing in ordered:
        certification=({"kind":"exact-grid","feasible":True,
                        "reason":"half-integer coordinates, zero rotations"}
                       if packing==baseline else _certify(n,packing))
        if certification["feasible"]:
            selected=packing
            break
        rejected.append(certification)
    side=validate(selected,n)
    return {"n":n,"packing":[[s.x,s.y,s.angle] for s in selected],
            "L":side,"grid_L":baseline_side,"improved":side<baseline_side-1e-7,
            "excess":side/math.sqrt(n)-1,"report":report,
            "certification":certification,"rejections":rejected,
            "total_seconds":time.perf_counter()-before}

def _atomic_json(path,value):
    temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")
    temporary.replace(path)

def _run_all(output,preset="quick",workers=2,resume=False,max_n=1000,
            saved="submissions/improved-puzzles-002.csv",
            metric="runs/official-metric/squares-in-squares-v17.ipynb",
            small_seconds=None,large_seconds=None,seed=1000):
    if preset not in ("quick","overnight") or not 1<=workers<=4 or not 1<=max_n<=1000:
        raise ValueError("invalid local-run configuration")
    settings={"preset":preset,"dense_max_n":30,
              "small_seconds":10.0 if preset=="quick" else 60.0,
              "large_seconds":2.0 if preset=="quick" else 40.0,
              "large_maxiter":18 if preset=="quick" else 60,
              "seed":seed,"max_n":max_n}
    if small_seconds is not None: settings["small_seconds"]=small_seconds
    if large_seconds is not None: settings["large_seconds"]=large_seconds
    if any(not math.isfinite(settings[k]) or settings[k]<=0
           for k in ("small_seconds","large_seconds")):
        raise ValueError("time budgets must be finite and positive")
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    checkpoint_dir=output/"puzzles"
    checkpoint_dir.mkdir(exist_ok=True)
    names=("local_run.py","scalable.py","warmstarts.py","basin.py",
           "optimize.py","geometry.py","submission.py","official.py","baseline.py")
    hashes={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in names}
    config={"settings":settings,"input_sha256":hashlib.sha256(Path(saved).read_bytes()).hexdigest(),
            "metric_sha256":hashlib.sha256(Path(metric).read_bytes()).hexdigest() if metric else None,
            "source_hashes":hashes}
    config_path=output/"settings.json"
    if config_path.exists():
        if not resume: raise FileExistsError("run exists; use --resume")
        if json.loads(config_path.read_text())!=config:
            raise ValueError("resume settings, input, metric, or algorithm sources differ")
    else:
        if resume: raise ValueError("no checkpoint to resume")
        _atomic_json(config_path,config)
        sources=output/"source"
        sources.mkdir(exist_ok=True)
        for name in names:
            (sources/name).write_bytes(Path(__file__).with_name(name).read_bytes())
        _atomic_json(output/"environment.json",{"python":platform.python_version(),
            "cpu_count":os.cpu_count(),"workers":workers,"network_calls":False,"llm_calls":False})
    if (output/"summary.json").exists():
        return json.loads((output/"summary.json").read_text())
    starts=extended_starts(read_submission(saved),max_n)
    completed={}
    for path in checkpoint_dir.glob("*.json"):
        result=json.loads(path.read_text())
        n=result["n"]
        if not 1<=n<=max_n: raise ValueError("checkpoint n out of range")
        actual=validate([Square(*row) for row in result["packing"]],n)
        if abs(actual-result["L"])>1e-9 or not result["certification"]["feasible"]:
            raise ValueError("invalid checkpoint")
        completed[n]=result
    started=time.perf_counter()
    stopped=False
    failures=[{"n":n,"error":r["report"]["error"]} for n,r in completed.items()
              if "error" in r["report"]]
    tasks=[(n,starts.get(n),settings) for n in range(1,max_n+1) if n not in completed]
    print(f"Local {preset} run: {len(completed)}/{max_n} complete; {workers} workers.",flush=True)
    _atomic_json(output/"progress.json",{"status":"running","completed":len(completed),
        "total":max_n,"pid":os.getpid(),"workers":workers})
    executor=concurrent.futures.ProcessPoolExecutor(max_workers=workers,
        initializer=_init_worker,initargs=(str(metric) if metric else None,))
    pending={executor.submit(_worker,task):task[0] for task in tasks}
    try:
        for future in concurrent.futures.as_completed(pending):
            if (output/"STOP").exists():
                stopped=True
                for item in pending: item.cancel()
                break
            n=pending[future]
            try:
                result=future.result()
            except Exception as error:
                fallback=grid(n)
                side=validate(fallback,n)
                result={"n":n,"packing":[[s.x,s.y,s.angle] for s in fallback],
                    "L":side,"grid_L":side,"improved":False,"excess":side/math.sqrt(n)-1,
                    "report":{"algorithm":"error-grid-fallback","error":repr(error)},
                    "certification":{"kind":"exact-grid","feasible":True},
                    "rejections":[],"total_seconds":0}
                failures.append({"n":n,"error":repr(error)})
            _atomic_json(checkpoint_dir/f"{n:04d}.json",result)
            completed[n]=result
            progress={"status":"running","completed":len(completed),"total":max_n,
                "improved_puzzles":sum(r["improved"] for r in completed.values()),
                "last_n":n,"last_L":result["L"],"session_seconds":time.perf_counter()-started,
                "pid":os.getpid(),"workers":workers,"failed_puzzles":failures}
            _atomic_json(output/"progress.json",progress)
            print(f"{len(completed)}/{max_n}: n={n}, L={result['L']:.9f}, "
                  f"improved={result['improved']}, method={result['report']['algorithm']}",flush=True)
    finally:
        executor.shutdown(wait=True,cancel_futures=True)
    if stopped:
        _atomic_json(output/"progress.json",{"status":"stopped","completed":len(completed),
            "total":max_n,"resume":"remove STOP and repeat with --resume"})
        return {"status":"stopped","completed":len(completed)}
    if len(completed)!=max_n: raise RuntimeError("missing puzzle checkpoint")
    packings={n:[Square(*row) for row in completed[n]["packing"]] for n in range(1,max_n+1)}
    submission=output/"submission.csv"
    temporary_submission=output/("submission-build-"+str(os.getpid())+".csv")
    write_submission(packings,temporary_submission)
    readback=read_submission(temporary_submission)
    scored=score(readback,max_n=max_n)
    rows=max_n*(max_n+1)//2
    if sum(map(len,readback.values()))!=rows: raise RuntimeError("wrong row count")
    temporary_submission.replace(submission)
    summary={"status":"complete","puzzles":max_n,"rows":rows,
        "score_estimate":scored["score"],"submission":str(submission),
        "improved_puzzles":sum(r["improved"] for r in completed.values()),
        "officially_checked_puzzles":sum(r["certification"]["kind"]=="official-notebook"
                                       for r in completed.values()),
        "exact_grid_puzzles":sum(r["certification"]["kind"]=="exact-grid"
                                for r in completed.values()),
        "failed_puzzles":failures,"session_seconds":time.perf_counter()-started,
        "submitted_to_kaggle":False}
    _atomic_json(output/"summary.json",summary)
    _atomic_json(output/"progress.json",summary)
    print(json.dumps(summary,indent=2),flush=True)
    return summary


def run_all(output,*args,**kwargs):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    lock=output/"active.lock"
    with lock.open("x",encoding="utf-8") as handle:
        json.dump({"pid":os.getpid()},handle)
    try:
        return _run_all(output,*args,**kwargs)
    except Exception as error:
        _atomic_json(output/"progress.json",{"status":"failed","error":repr(error),
                     "resume":"inspect the error, then rerun with --resume"})
        raise
    finally:
        lock.unlink(missing_ok=True)
