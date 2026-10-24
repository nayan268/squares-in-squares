"""Adapter for a separately downloaded official metric notebook.

Only load notebook code you trust. Source and provenance are retained locally.
"""
import __future__
import csv
import hashlib
import json
import sys
import types
from pathlib import Path


def load_metric(path):
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix==".ipynb":
        nb = json.loads(raw)
        source = "\n\n".join("".join(c["source"]) for c in nb["cells"]
                              if c["cell_type"]=="code")
    else:
        source = raw.decode("utf-8-sig")
    name = "_squares_official_" + hashlib.sha256(raw).hexdigest()[:12]
    module = types.ModuleType(name)
    sys.modules[name] = module
    # Postponing annotations lets the original Python 3.10+ hints load on 3.9.
    code = compile(source,str(path),"exec",flags=__future__.annotations.compiler_flag)
    exec(code,module.__dict__)
    return module, {"source_sha256":hashlib.sha256(raw).hexdigest(),
                    "source_file":str(path),
                    "adapter":"original verify_puzzle, measure_side_mp and score_puzzle; postponed annotations"}


def check_submission(path, metric_path):
    metric, provenance = load_metric(metric_path)
    grouped = {}
    seen = set()
    with Path(path).open(newline="",encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if int(row["id"]) in seen:
                raise ValueError("duplicate row id")
            seen.add(int(row["id"]))
            grouped.setdefault(int(row["n"]),[]).append(row)
    if not grouped or any(not 1<=n<=1000 for n in grouped):
        raise ValueError("no puzzles or n out of range")
    results=[]
    for n,rows in sorted(grouped.items()):
        error=metric._validate_puzzle(n,rows)
        if error:
            raise ValueError(error)
        squares,side=metric._build_squares(n,rows)
        feasible,reason=metric.verify_puzzle(squares)
        results.append({"n":n,"feasible":feasible,"reason":reason,
                        "L":metric.mpmath.nstr(side,55) if feasible else None,
                        "score":metric.score_puzzle(n,float(side)) if feasible else metric.infeasible_score(n)})
    total=sum(row["score"] for row in results)
    total+=sum(metric.infeasible_score(n) for n in range(1,1001) if n not in grouped)
    return {"provenance":provenance,"puzzles":results,
            "all_feasible":all(row["feasible"] for row in results),
            "competition_score":total}
