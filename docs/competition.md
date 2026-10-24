# Competition contract

Verified against public pages on 2026-10-24:

- [Overview and evaluation](https://www.kaggle.com/competitions/squares-in-squares/overview)
- [Official metric notebook](https://www.kaggle.com/code/metric/squares-in-squares)
- [Data](https://www.kaggle.com/competitions/squares-in-squares/data)
- [Rules](https://www.kaggle.com/competitions/squares-in-squares/rules)

There are 1,000 puzzles: n=1 through 1,000. Each consists of n unit squares
specified by centers and rotation angles in radians. Interiors must not
intersect; boundary contacts are allowed. The container is axis aligned.

L = max(max_corner_x - min_corner_x, max_corner_y - min_corner_y).
This measurement is translation invariant.

The CSV header is id,n,square_id,x,y,angle. Row id is a unique integer; this
baseline uses consecutive integers globally and square_id starts at zero within
each puzzle. Coordinates are decimal literals, optionally scientific notation.
No s prefixes or symbolic expressions. L is computed, not submitted. A full
submission contains 500,500 rows.

Lower is better: sum L / sqrt(n) - 1 across the puzzle universe. Missing and
invalid puzzles receive the grid ceiling ceil(sqrt(n)) / sqrt(n) - 1.
Partial submissions are allowed. The local tool rejects invalid packings
instead of silently applying this fallback.

The official notebook uses mpmath interval arithmetic at escalating precision
to certify feasibility and high precision to measure L. The local Python float
SAT checker is a development screen, not that certificate. It may disagree on
nearly touching rotated squares or very large coordinates. It does not preserve
arbitrary decimal precision. The baseline uses exact half-integers and angle 0.

The official metric and rules remain authoritative. Verify updates before
submitting. This repo includes no competition data or accepted competition terms. The
first optimization run retains an untracked local official metric notebook;
see [optimization.md](optimization.md) for its provenance and adapter.
