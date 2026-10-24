import argparse
import json
from pathlib import Path
from .baseline import grid
from .submission import read_submission, write_submission, score


def main(argv=None):
    parser = argparse.ArgumentParser(description="Squares in Squares experiment tools")
    commands = parser.add_subparsers(dest="command", required=True)
    baseline = commands.add_parser("baseline", help="write deterministic grid CSV")
    baseline.add_argument("--max-n", type=int, default=1000)
    baseline.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("validate", help="screen geometry and estimate score")
    check.add_argument("submission", type=Path)
    check.add_argument("--max-n", type=int, default=1000,
                       help="scored universe; leave at 1000 for competition estimates")
    check.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "baseline":
            if not 1 <= args.max_n <= 1000:
                raise ValueError("max-n must be from 1 through 1000")
            packings = {n: grid(n) for n in range(1, args.max_n + 1)}
            write_submission(packings, args.output)
            print("Wrote {} puzzles ({} rows) to {}".format(
                len(packings), sum(map(len, packings.values())), args.output))
        else:
            result = score(read_submission(args.submission), args.max_n)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                with args.report.open("x", encoding="utf-8") as handle:
                    json.dump(result, handle, indent=2, allow_nan=False)
            print(json.dumps({key: value for key, value in result.items()
                              if key != "details"}, indent=2))
            print("Local float screening passed; use the official metric for certification.")
    except (ValueError, OSError) as error:
        parser.exit(2, "error: {}\n".format(error))
