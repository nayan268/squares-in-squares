# Local all-puzzle run

The optimization runs in standalone Python processes on this laptop. The job
makes no LLM/API calls and no network requests. Kaggle upload is a separate step
performed only after the complete CSV has been validated.

## Current quick pass

~~~powershell
./scripts/run-local.ps1 -Preset quick -Workers 2 -RunName local-all-001 -Wait
~~~

Two workers each use one numerical-library thread. The launcher requests below
normal priority and keeps a supervising local session open with -Wait.
Keep the laptop awake and plugged in until completion. Do not start a second
copy of the same run.

Every n=1..1000 is covered. Perfect-square grids already meet the area lower
bound, so those return immediately. For other puzzles:

- n <= 30: the existing analytic basin-hopping solver, 10 seconds per puzzle;
- n > 30: a memory-bounded penalty basin variant, 2 seconds per puzzle;
- starts: saved certified packings, plus valid extensions with unit-square border strips;
- fallbacks: the best saved valid packing or the exact grid.

The larger-puzzle path directly accumulates gradients. It never allocates the
dense pair-constraint Jacobian, which would be impractical at n=1000. Pair
screening uses the maximum possible center distance of overlapping unit squares.
After local refinement, uniform center dilation repairs residual overlaps; the
result is independently validated before it can replace an incumbent.

The search is heuristic. Short budgets can end before a completed hop, and some
puzzles will retain their grid fallback. Coverage of all 1,000 puzzles does not
mean every packing has improved.

The quick pass is intended to take roughly 30â€“60 minutes, including initialization,
checking and export; actual time depends on the laptop. The overnight preset
uses 60 seconds for small puzzles and 40 seconds for larger puzzles.

## Progress and output

~~~powershell
Get-Content runs/local-all-001/progress.json
Get-Content runs/local-all-001/console-*.log -Tail 10
~~~

The final file is runs/local-all-001/submission.csv. A complete run has 1,000
puzzles and 500,500 rows. summary.json records the score estimate, coverage,
certification counts and failed puzzles. Each completed puzzle has a separate
JSON checkpoint under puzzles/.

Improved non-grid packings are checked with the downloaded official metric's
original feasibility functions. Exact grid fallbacks use half-integer centers
and zero rotations. The final full CSV is read back and locally validated.

## Stop and resume

To stop after currently running work:

~~~powershell
New-Item -ItemType File runs/local-all-001/STOP
~~~

After the process has stopped, resume with the same settings:

~~~powershell
./scripts/run-local.ps1 -Preset quick -Workers 2 -RunName local-all-001 -Resume -Wait
~~~

The launcher removes the STOP marker when explicitly resuming. Resume validates
the algorithm source hashes, saved input, metric and search settings before
reusing checkpoints. An active.lock prevents duplicate runs. If the laptop or
process crashes, verify the old process is gone before manually removing that
specific stale lock file. Do not clear a lock belonging to a running job.

The source snapshots and settings remain in the run folder. No Kaggle credentials
are stored there, and the runner never submits anything by itself.
