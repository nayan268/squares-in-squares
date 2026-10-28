import tempfile
import unittest
from pathlib import Path
from squares.baseline import grid
from squares.submission import read_submission, write_submission, score


class SubmissionTests(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "submission.csv"
            original = {n: grid(n) for n in range(1, 11)}
            write_submission(original, path)
            self.assertEqual(original, read_submission(path))
            with self.assertRaises(FileExistsError):
                write_submission(original, path)

    def test_missing_puzzles_receive_grid_penalty(self):
        self.assertEqual(score({1: grid(1)})["score"], score({})["score"])
        self.assertEqual(score({4: grid(4)}, max_n=4)["details"][-1]["excess"], 0)

    def test_invalid_csvs(self):
        invalid_rows = [
            "0,2,0,0.5,0.5,0\n",
            "0,2,0,0.5,0.5,0\n0,2,1,1.5,0.5,0\n",
            "0,2,0,0.5,0.5,0\n1,2,0,1.5,0.5,0\n",
            "0,1,0,s0.5,0.5,0\n",
            "0,1,0,nan,0.5,0\n",
            "0,1,0,1e999,0.5,0\n",
            "0,1,0,0.5,0.5,0,extra\n",
            "0,2,0,0.5,0.5,0\n1,2,1,0.5,0.5,0\n",
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            for rows in invalid_rows:
                with self.subTest(rows=rows):
                    path.write_text("id,n,square_id,x,y,angle\n" + rows, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        read_submission(path)


if __name__ == "__main__":
    unittest.main()
