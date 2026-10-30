"""Build and certify a complete submission from saved local artifacts."""
import json
from pathlib import Path

from .geometry import validate
from .official import check_submission, load_metric
from .research_seeds import propagated_seeds
from .structured_experiment import _atomic, _certify
from .submission import read_submission, score, write_submission


def build_candidate(output, initial_path, partial_paths=(), metric_path=None,
                    max_n=1000):
    """Merge only strictly better, certified packings into a full candidate."""
    output=Path(output)
    if output.exists():
        raise FileExistsError("candidate output already exists")
    output.mkdir(parents=True)
    initial_path=Path(initial_path)
    partial_paths=tuple(map(Path,partial_paths))
    metric_path=Path(metric_path) if metric_path else None
    initial=read_submission(initial_path)
    expected=set(range(1,max_n+1))
    if set(initial)!=expected:
        raise ValueError("initial submission must contain every requested puzzle")
    candidates={n:[] for n in expected}
    for path in partial_paths:
        for n,packing in read_submission(path).items():
            if n<=max_n:
                candidates[n].append((str(path),packing))
    for n,packing in propagated_seeds(max_n).items():
        candidates[n].append(("published-border-family",packing))

    metric=load_metric(metric_path)[0] if metric_path else None
    merged={n:list(initial[n]) for n in expected}
    changes=[]
    rejected=[]
    for n in range(1,max_n+1):
        original_side=validate(initial[n],n)
        options=sorted(candidates[n],key=lambda item:validate(item[1],n))
        for source,packing in options:
            side=validate(packing,n)
            if side>=validate(merged[n],n)-1e-10:
                continue
            certification=_certify(metric,n,packing)
            if not certification["feasible"]:
                rejected.append({"n":n,"source":source,"L":side,
                                 "certification":certification})
                continue
            merged[n]=list(packing)
            changes.append({"n":n,"source":source,"original_L":original_side,
                            "new_L":side,
                            "score_gain":(original_side-side)/(n**0.5),
                            "certification":certification})
            break

    csv_path=output/"submission.csv"
    write_submission(merged,csv_path)
    readback=read_submission(csv_path)
    original_score=score(initial,max_n)["score"]
    candidate_score=score(readback,max_n)["score"]
    official=None
    if metric_path:
        official=check_submission(csv_path,metric_path)
        _atomic(output/"official-report.json",official)
        if not official["all_feasible"]:
            raise RuntimeError("official metric rejected the merged submission")
    report={"status":"complete","max_n":max_n,
            "initial":str(initial_path),"partials":list(map(str,partial_paths)),
            "submission":str(csv_path),"changed_puzzles":len(changes),
            "local_original_score":original_score,
            "local_candidate_score":candidate_score,
            "local_score_gain":original_score-candidate_score,
            "official_all_feasible":official["all_feasible"] if official else None,
            "official_competition_score":official["competition_score"] if official else None,
            "changes":changes,"rejected_candidates":rejected}
    _atomic(output/"summary.json",report)
    return report
