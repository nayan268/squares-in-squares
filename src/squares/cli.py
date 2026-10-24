import argparse
import json
from pathlib import Path
from .baseline import grid
from .submission import read_submission, write_submission, score


def main(argv=None):
    parser=argparse.ArgumentParser(description="Squares in Squares experiment tools")
    commands=parser.add_subparsers(dest="command",required=True)
    baseline=commands.add_parser("baseline",help="write deterministic grid CSV")
    baseline.add_argument("--max-n",type=int,default=1000)
    baseline.add_argument("--output",type=Path,required=True)
    check=commands.add_parser("validate",help="screen geometry and estimate score")
    check.add_argument("submission",type=Path)
    check.add_argument("--max-n",type=int,default=1000)
    check.add_argument("--report",type=Path)
    reference=commands.add_parser("reference",help="export the known five-square construction")
    reference.add_argument("--output",type=Path,required=True)
    plot=commands.add_parser("plot",help="draw a puzzle from a valid submission")
    plot.add_argument("submission",type=Path)
    plot.add_argument("--n",type=int,required=True)
    plot.add_argument("--output",type=Path,required=True)
    official=commands.add_parser("official-check",help="check with a trusted downloaded metric")
    official.add_argument("submission",type=Path)
    official.add_argument("--metric",type=Path,required=True)
    official.add_argument("--report",type=Path)
    for command in ("optimize","benchmark"):
        p=commands.add_parser(command,help="run the multistart solver" if command=="optimize"
                              else "compare independent seeds, keeping references separate")
        p.add_argument("--output",type=Path,required=True)
        p.add_argument("--restarts",type=int,default=6)
        p.add_argument("--maxiter",type=int,default=150)
        p.add_argument("--squeeze-steps",type=int,default=4)
        if command=="optimize":
            p.add_argument("--n",type=int,required=True)
            p.add_argument("--seed",type=int,default=0)
            p.add_argument("--report",type=Path)
        else:
            p.add_argument("--cases",type=int,nargs="+",default=[5,10,11])
            p.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
            p.add_argument("--metric",type=Path)
    basin=commands.add_parser("basin",help="run basin hopping with analytic local refinement")
    basin.add_argument("--n",type=int,default=11)
    basin.add_argument("--seed",type=int,default=0)
    basin.add_argument("--seconds",type=float,default=60)
    basin.add_argument("--hops",type=int,default=1000)
    basin.add_argument("--maxiter",type=int,default=140)
    basin.add_argument("--temperature",type=float,default=0.04)
    basin.add_argument("--output",type=Path,required=True)
    basin.add_argument("--report",type=Path)
    comparison=commands.add_parser("basin-benchmark",help="equal-time comparison with the original solver")
    comparison.add_argument("--n",type=int,default=11)
    comparison.add_argument("--seeds",type=int,nargs="+",default=[20,21,22])
    comparison.add_argument("--seconds",type=float,default=60)
    comparison.add_argument("--output",type=Path,required=True)
    comparison.add_argument("--metric",type=Path)
    comparison.add_argument("--resume",action="store_true")
    args=parser.parse_args(argv)
    try:
        if args.command=="baseline":
            if not 1<=args.max_n<=1000:
                raise ValueError("max-n must be from 1 through 1000")
            packings={n:grid(n) for n in range(1,args.max_n+1)}
            write_submission(packings,args.output)
            print("Wrote {} puzzles ({} rows) to {}".format(
                len(packings),sum(map(len,packings.values())),args.output))
        elif args.command=="reference":
            from .constructions import five_square_reference
            write_submission({5:five_square_reference()},args.output)
            print("Wrote reference:",args.output)
        elif args.command=="plot":
            from .plot import plot_packings
            packings=read_submission(args.submission)
            if args.n not in packings:
                raise ValueError("requested n is missing from submission")
            plot_packings([("Packing",packings[args.n])],args.output)
            print("Wrote plot:",args.output)
        elif args.command=="basin":
            from .basin import solve_basin,BasinConfig
            if args.output.exists() or (args.report and args.report.exists()):
                raise FileExistsError("choose fresh output and report paths")
            packing,result=solve_basin(args.n,BasinConfig(seed=args.seed,seconds=args.seconds,
                                        hops=args.hops,maxiter=args.maxiter,
                                        temperature=args.temperature))
            write_submission({args.n:packing},args.output)
            if args.report:
                _report(args.report,result)
            print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2))
        elif args.command=="basin-benchmark":
            from .basin_experiment import compare
            print(json.dumps(compare(args.output,args.n,args.seeds,args.seconds,args.metric,args.resume),indent=2))
        elif args.command=="optimize":
            from .optimize import solve,SolverConfig
            packing,result=solve(args.n,SolverConfig(seed=args.seed,restarts=args.restarts,
                                 maxiter=args.maxiter,squeeze_steps=args.squeeze_steps))
            write_submission({args.n:packing},args.output)
            if args.report:
                _report(args.report,result)
            print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2))
        elif args.command=="benchmark":
            from .experiment import benchmark
            from .optimize import SolverConfig
            benchmark(args.output,args.cases,args.seeds,
                      SolverConfig(restarts=args.restarts,maxiter=args.maxiter,
                                   squeeze_steps=args.squeeze_steps),args.metric)
        elif args.command=="official-check":
            from .official import check_submission
            result=check_submission(args.submission,args.metric)
            if args.report:
                _report(args.report,result)
            print(json.dumps(result,indent=2))
            if not result["all_feasible"]:
                parser.exit(2,"Official feasibility check rejected one or more puzzles.\n")
        else:
            result=score(read_submission(args.submission),args.max_n)
            if args.report:
                _report(args.report,result)
            print(json.dumps({k:v for k,v in result.items() if k!="details"},indent=2))
            print("Local float screening passed; use the official metric for certification.")
    except (ValueError,OSError,RuntimeError) as error:
        parser.exit(2,"error: {}\n".format(error))


def _report(path,result):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
