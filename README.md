# Squares in Squares

A Python project for [Kaggle Squares in Squares](https://www.kaggle.com/competitions/squares-in-squares):
pack n unit squares into the smallest axis-aligned square for n=1..1000.

Includes a deterministic grid baseline, rotated-square geometry, spatially
bucketed overlap checks, official-format CSV I/O, score estimates, tests, and
GitHub Actions. Core tools need only the standard library. A small-n multistart solver,
reference construction, plotter, and reproducible benchmark workflow are included.
See [the optimization guide](docs/optimization.md).

## Quick start (PowerShell)

Requires Python 3.9+ and Git. From this repository directory:

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\squares.exe baseline --max-n 1000 --output submissions/grid-new.csv
.\.venv\Scripts\squares.exe validate submissions/grid-new.csv --report runs/grid-new-report.json
~~~

On macOS/Linux use .venv/bin/python and .venv/bin/squares instead.
Output files are never overwritten; choose a new name for each experiment.

A prepared baseline is included locally as submissions/grid.csv with its
report at runs/grid-report.json. These generated files are ignored by Git.
Use a different output name when regenerating them.

For a quick partial submission, use --max-n 10 in the baseline command.
Validation still scores all 1,000 puzzles by default, charging missing puzzles
the grid ceiling. Its --max-n option changes the scoring universe for small
tests; those results are not comparable with the full competition score.

Without installing the package, run from the repo in PowerShell:

~~~powershell
$env:PYTHONPATH = "$PWD/src"
python -m squares --help
python -m unittest discover -s tests -v
~~~

## Kaggle access

No data or credentials are needed for the baseline. To download official files,
join the competition and accept its rules on Kaggle, then install the optional
CLI and authenticate using [Kaggle's official instructions](https://github.com/Kaggle/kaggle-api#authentication).
Keep credentials outside this repo.

~~~powershell
.\.venv\Scripts\python.exe -m pip install -e '.[kaggle]'
.\.venv\Scripts\kaggle.exe competitions download -c squares-in-squares -p data/raw
~~~

After validating with the official metric, a manual submission command is:

~~~powershell
.\.venv\Scripts\kaggle.exe competitions submit -c squares-in-squares -f submissions/grid.csv -m 'Grid baseline'
~~~

This setup does not join the competition or submit automatically.

## Layout

| Path | Purpose |
| --- | --- |
| src/squares/ | Geometry, baseline, CSV handling, CLI |
| tests/ | Geometry and submission checks |
| data/raw/ | Untracked official downloads |
| runs/ | Untracked experiment reports |
| submissions/ | Untracked generated CSVs |
| notebooks/ | Exploratory work |
| docs/competition.md | Verified format, objective, limitations |
| docs/experiments.md | Solver extension and experiment workflow |

The estimate sums L / sqrt(n) - 1; lower is better. The local validator uses
floats, while Kaggle certifies feasibility with interval arithmetic. Use the
[official metric notebook](https://www.kaggle.com/code/metric/squares-in-squares)
for final checks, particularly for nearly touching rotated squares.

## Developing a solver

~~~python
from squares import grid, validate
from squares.submission import write_submission

packings = {n: grid(n) for n in range(1, 11)}
# Replace entries with improved lists of Square(x, y, angle_in_radians).
for n, packing in packings.items():
    validate(packing, expected_n=n)
write_submission(packings, "submissions/experiment-001.csv")
~~~

## Git

The repo starts on codex/project-setup with an initial scaffold commit.
No remote is configured. Create a private repository in your preferred Git host:

~~~powershell
git remote add origin <your-private-repository-url>
git push -u origin codex/project-setup
~~~

## First optimization run

See [docs/optimization.md](docs/optimization.md) for the solver, plot commands,
benchmark settings, and official metric adapter. The first local results are
under runs/benchmark-001/; generated files remain ignored by Git.

[First benchmark results](docs/first-results.md) include measured outcomes,
per-seed success rates, and the next experiment.

## Basin hopping for eleven squares

The new solver supports collective moves, penalized exploration states,
analytic local derivatives, and a wall-clock budget.
See [the basin-hopping guide](docs/basin-hopping.md) for commands and evaluation.

~~~powershell
$env:PYTHONPATH = "$PWD/src"
python -m squares basin --n 11 --seed 20 --seconds 60 --output submissions/basin-new.csv --report runs/basin-new.json
~~~

[Eleven-square basin-hopping results](docs/basin-results.md) record the
matched-budget comparison and updated incumbent.

## Structured maximal-inflation experiments

The research-inspired solver works in a fixed unit container, inflates all
squares until contacts form, and tries diagonal-band, rotated-block, staggered-row,
tilted-strip, and incumbent-perturbation templates. Runs are local, resumable,
and never call a network service or language model.

~~~powershell
$env:PYTHONPATH = "$PWD/src"
python -m squares structured-benchmark `
  --cases 11 17 18 19 29 --seeds 0 1 2 --seconds 30 `
  --initial runs/local-all-001/submission.csv `
  --metric runs/official-metric/squares-in-squares-v17.ipynb `
  --output runs/structured-001
~~~

If interrupted, repeat the same command with `--resume`. Each case/seed is
checkpointed separately, and `best-cases.csv` contains only the requested cases.

## Balanced row-family pilot

Cases just above a perfect square use two complementary mechanisms. Exact
border propagation carries a verified packing from `k^2+r` to
`(k+1)^2+r`; the low-dimensional refinement searches balanced row counts,
global tilt, alternating row angles, spacing, and row offsets. It is a local
CPU search and does not call a network service or a language model.

~~~powershell
$env:PYTHONPATH = "$PWD/src"
python -m squares family-benchmark `
  --cases 28 31 39 42 52 55 67 70 84 87 `
  --seeds 101 202 303 --seconds 20 `
  --initial runs/local-all-001/submission.csv `
  --metric runs/official-metric/squares-in-squares-v17.ipynb `
  --output runs/family-001
~~~

Repeat the exact command with `--resume` after an interruption. Outputs are
checkpointed once per case and seed; the resulting CSV is intentionally only a
selected-case artifact, not a complete Kaggle submission.

## Full local pass

See [the all-puzzle local-run guide](docs/local-all-run.md) for the standalone
CPU job, progress logs, checkpoints, stop/resume commands, and validation scope.
