# Basin hopping for eleven squares

Implemented 2026-10-25. This is a custom basin-hopping loop using SciPy local
minimizers, with explicit validity checks, a wall-clock deadline, and a separate
best feasible incumbent.

## Reproduce

Install the optional solver, plotting, and metric dependencies described in
[optimization.md](optimization.md). From the repository directory:

~~~powershell
$env:PYTHONPATH = "$PWD/src"
python -m squares basin --n 11 --seed 20 --seconds 60 --output submissions/basin-11-new.csv --report runs/basin-11-new.json
python -m squares official-check submissions/basin-11-new.csv --metric runs/official-metric/squares-in-squares-v17.ipynb --report runs/basin-11-new-official.json
python -m squares basin-benchmark --n 11 --seeds 20 21 22 --seconds 60 --metric runs/official-metric/squares-in-squares-v17.ipynb --output runs/basin-benchmark-002
~~~

Choose fresh output paths. The downloaded metric notebook stays ignored by Git;
a fresh clone must download its own trusted copy as described in the guide.

## Algorithm

The search alternates between a proposal and local refinement:

1. Propose a collective rotation, tilt a group, reinsert a group, expand and
   shake, or generate a random lattice/pose restart. Translation amplitudes
   cycle through 0.10, 0.25, and 0.45.
2. Minimize L plus a squared violation penalty with L-BFGS-B, increasing the
   penalty weights through 40, 400, and 4000.
3. Refine L with explicit wall and SAT separation constraints using SLSQP.
4. Accept or reject the refined search state using the Metropolis rule:
   accept lower energy, otherwise accept with probability exp(-delta/T),
   with T=0.04 by default.
5. Independently validate every feasible candidate before it can replace
   the best saved packing.

The active search state may be infeasible. Its energy is L plus the 4000-weight
squared violation penalty. This permits exploration through a slightly
overlapping intermediate arrangement. A feasible candidate's walk energy is
its measured, safety-expanded side length. Invalid states are never exported
or substituted for the best valid incumbent. Reports distinguish candidate
validity, walk energy, accepted uphill moves, and best valid L.

During search, centers may range up to grid L + 0.25. This permits temporary
worse layouts while preserving the original grid as a valid fallback.

Analytic derivatives cover the wall margins and the active SAT separating
axis for every pair. At absolute-value zeros or tied axes, the code uses a
selected subgradient. The problem remains nonsmooth and nonconvex; analytic
derivatives do not guarantee a better packing.

The best valid packing starts at the grid. No reference packing is imported
into the search. A small center expansion protects exported decimal geometry
from rounding. Every benchmark CSV is checked using the downloaded official
metric's original functions before being accepted into the comparison's best
output.

## Evaluation

Development pilots used seeds 0 and 1. The first valid-state-only pilot stayed
at L=4 after 20 seconds. The second pilot, using penalized exploration states,
found L=3.925276029734657 in 25 seconds; its exported CSV passed the official
metric's feasibility functions. These are development results, not held-out
benchmark evidence.

The comparison uses separate seeds 20, 21, and 22, with 60 seconds per method
per seed. The original solver gets the same deadline and enough restarts to
use its budget. Method order alternates between seeds. Metric checking is
outside the solver timing. Source snapshots, hashes, parameters, histories,
and official check reports are saved under runs/basin-benchmark-001/.

This comparison evaluates the complete new pipeline (basin hopping and
analytic derivatives) against the original finite-difference multistart
pipeline. It does not isolate which change causes any gain. Three seeds are
a small preliminary sample, and equal time rather than equal hop counts is
the relevant budget here.

## Tests and limitations

The new tests check analytic derivatives against central finite differences,
agreement with the previous SAT margins, Metropolis acceptance, repeatable
group moves, deadline fallback, perfect-square controls, and preservation of
a valid incumbent when an invalid exploration state is accepted.

The new tests live in tests/test_basin.py. Run the full test suite after
installing the optional solver and plotting dependencies.

Basin hopping is stochastic and offers no global optimum certificate.
Published layouts provide a numerical target around 3.877083590 for n=11;
that layout is not used to seed these experiments.
[Reference catalogue](https://kingbird.myphotos.cc/packing/squares_in_squares.html).
The general perturb/refine/accept framework is described in
[SciPy's basin-hopping documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html).

## Resuming comparisons

Repeat the same basin-benchmark command with --resume and the same output
directory to reuse completed seed runs. The runner verifies the puzzle, seeds,
time budget, metric source, and saved solver-source hashes. Changes to the
comparison writer itself are recorded separately in resume provenance.
