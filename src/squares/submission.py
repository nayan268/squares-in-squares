"""Official CSV columns, with strict local rejection of invalid packings."""
import csv
import math
import re
from pathlib import Path
from .geometry import Square, validate

COLUMNS = ["id", "n", "square_id", "x", "y", "angle"]
DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
INTEGER = re.compile(r"[0-9]+\Z")


def write_submission(packings, path):
    if not packings:
        raise ValueError("packings must not be empty")
    for n, squares in packings.items():
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 1000:
            raise ValueError("n must be an integer from 1 through 1000")
        validate(squares, n)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        row_id = 0
        for n, squares in sorted(packings.items()):
            for square_id, square in enumerate(squares):
                writer.writerow([row_id, n, square_id,
                                 repr(float(square.x)), repr(float(square.y)),
                                 repr(float(square.angle))])
                row_id += 1


def read_submission(path):
    grouped, seen = {}, set()
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise ValueError("expected CSV header: " + ",".join(COLUMNS))
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError("wrong column count on line {}".format(line))
            if not all(INTEGER.fullmatch(row[key]) for key in COLUMNS[:3]):
                raise ValueError("invalid integer on line {}".format(line))
            row_id, n, square_id = (int(row[key]) for key in COLUMNS[:3])
            if row_id in seen:
                raise ValueError("duplicate row id {}".format(row_id))
            seen.add(row_id)
            if not 1 <= n <= 1000 or not 0 <= square_id < n:
                raise ValueError("n or square_id out of range on line {}".format(line))
            if not all(DECIMAL.fullmatch(row[key]) for key in COLUMNS[3:]):
                raise ValueError("coordinates must be decimal literals")
            squares = grouped.setdefault(n, {})
            if square_id in squares:
                raise ValueError("duplicate square_id for n={}".format(n))
            squares[square_id] = Square(*(float(row[key]) for key in COLUMNS[3:]))
    if not grouped:
        raise ValueError("submission contains no puzzles")
    packings = {}
    for n, squares in grouped.items():
        if set(squares) != set(range(n)):
            raise ValueError("puzzle {} requires square_id 0 through {}".format(n, n - 1))
        packings[n] = [squares[i] for i in range(n)]
        validate(packings[n], n)
    return packings


def score(packings, max_n=1000):
    """Local estimate; missing puzzles receive the grid ceiling.

    Invalid packings raise rather than receiving the official fallback.
    Float screening cannot certify close contacts like interval arithmetic.
    """
    if not 1 <= max_n <= 1000 or any(n < 1 or n > max_n for n in packings):
        raise ValueError("puzzles must be within 1..max_n, max_n <= 1000")
    details = []
    for n in range(1, max_n + 1):
        present = n in packings
        side = validate(packings[n], n) if present else math.isqrt(n - 1) + 1
        details.append({"n": n, "L": side, "excess": side / math.sqrt(n) - 1,
                        "submitted": present})
    return {"score": math.fsum(row["excess"] for row in details),
            "submitted_puzzles": len(packings), "scored_puzzles": max_n,
            "details": details}
