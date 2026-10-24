# First optimization workflow

The solver starts independently from grids, randomized occupied lattice cells,
random centers/angles, and perturbed incumbents. The analytical construction is
a separate reference and is never passed to benchmark solver starts.

## Install and run

From the repository directory (PowerShell):

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[optimize,plot,official]'
.\.venv\Scripts\squares.exe reference --output submissions/five-reference.csv
.\.venv\Scripts\squares.exe plot submissions/five-reference.csv --n 5 --output runs/five-reference.svg
.\.venv\Scripts\squares.exe optimize --n 5 --seed 0 --restarts 6 --output submissions/five-search.csv --report runs/five-search.json
~~~

SVG plotting needs no additional dependencies. PNG plotting uses Pillow.
For source-only use, set $env:PYTHONPATH = "$PWD/src" and use python -m squares.

## Official metric

The first local run downloaded version 17 of the public
[metric notebook](https://www.kaggle.com/code/metric/squares-in-squares), using
its Download .ipynb action. The local copy is:

runs/official-metric/squares-in-squares-v17.ipynb

The downloaded notebook SHA-256 is
5c42955b6347e721dccaf72abec83f9e88fd488e3b9c40c1d23f476dc7189d94.
The notebook declares Apache 2.0. It is retained as an untracked downloaded
artifact; a fresh clone must download its own trusted copy. Never point the
adapter at untrusted code: it executes the notebook's Python cells.

~~~powershell
.\.venv\Scripts\squares.exe official-check submissions/five-search.csv --metric runs/official-metric/squares-in-squares-v17.ipynb --report runs/five-official.json
.\.venv\Scripts\squares.exe benchmark --cases 5 10 11 --seeds 0 1 2 --restarts 4 --maxiter 150 --squeeze-steps 3 --metric runs/official-metric/squares-in-squares-v17.ipynb --output runs/benchmark-002
~~~

Use a new output name for each run. Benchmark directories are never reused.
The adapter calls the original notebook's coordinate parsing, verify_puzzle,
measure_side_mp and score_puzzle functions. It postpones type annotations to
support Python 3.9; it does not alter geometric checks. It uses mpmath.nstr for
JSON reporting instead of the notebook's .man/.exp-based 50-place CSV formatter,
which requires a newer mpmath API. It does not call Kaggle or submit a score.
The notebook's main score() entry point is not used.

Each benchmark saves input settings, source hashes, library versions,
per-seed histories, CSVs, independent official feasibility reports, diagrams,
and a summary. Reported competition scores include the grid penalties for
missing puzzles. Keep improvement success separate from hitting the n=5
reference within 1e-5.

## Search mechanics

For each pair, the separating-axis theorem supplies four projected gaps.
The maximum gap must be nonnegative, so at least one axis separates the pair.
Wall margins keep all corners inside [0,L] squared.

A squeeze step decreases L, then minimizes squared negative margins using
L-BFGS-B. SLSQP repairs the constraints. Finally SLSQP also optimizes L.
A small positive clearance protects against serialization rounding.

Optimizer success flags are diagnostics, not feasibility certificates. A
candidate is independently checked with the original float geometry before
replacing the incumbent; benchmarks optionally recheck the exported decimal
CSV with the official notebook before accepting it into best.csv.

The n=5 mathematical reference is L=2+1/sqrt(2). Its four corner squares
surround a central 45-degree square. Reference coordinates are expanded very
slightly to make decimal exports strictly separated.
[Construction reference](https://erich-friedman.github.io/papers/squares/squares.html).

## Limits and next experiments

This is a small-n research solver, not a competitive large-n implementation.
Constraints evaluate all pairs and use numerical derivatives; runtime grows
quickly. A local minimum can leave the grid unbeaten. Passing the metric does
not prove optimality.

After the first benchmark, improve start diversity and contact rearrangements,
then try neighboring n with warm starts. Preserve the best certified packing for
each n. Perfect-square grids already attain the area bound, so the solver
returns them without searching.
