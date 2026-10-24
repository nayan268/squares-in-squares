# Experiment workflow

1. Start from squares.baseline.grid(n) or a validated incumbent.
2. Return a list of Square(x, y, angle) for each n; angles are radians.
3. Screen candidates with validate(candidate, expected_n=n). Use the official
   metric for near contacts before replacing a trusted incumbent.
4. Compare container_side(candidate) to the previous side length for that n.
5. Retain only independently validated improvements and combine puzzle results.
6. Export using write_submission and validate the exported CSV.

Next solvers could use simulated annealing on centers and angles, soft overlap
penalties with local optimization, and warm starts from adjacent n. Optional
dependencies: python -m pip install -e '.[optimize]'.

Record solver revision, parameters, random seed, n range, runtime, side lengths,
local and official scores, and rejected overlaps under runs/<unique-run-name>/.
The baseline has no random component. No competitive optimizer is implemented.
