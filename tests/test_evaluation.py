"""Tests for src/evaluation.py. Run with: python -m unittest discover tests"""
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.baselines import GlobalMean, MovieMean, Popularity
from src.evaluation import evaluate, run_experiments, save_results

TRAIN = pd.DataFrame(
    {
        "userId": [1, 1, 2, 2, 3, 3],
        "movieId": [10, 20, 10, 30, 10, 40],
        "rating": [4.0, 2.0, 5.0, 5.0, 3.0, 5.0],
    }
)
TEST = pd.DataFrame(
    {
        "userId": [1, 2, 3],
        "movieId": [30, 40, 30],
        "rating": [5.0, 4.0, 3.0],
    }
)


class TestEvaluate(unittest.TestCase):
    def test_rating_model_gets_all_metrics(self):
        row = evaluate(MovieMean(), TRAIN, TEST, k=1)
        # movie means: 30 -> 5.0, 40 -> 5.0; errors 0, 1, 2
        self.assertAlmostEqual(row["mae"], 1.0)
        self.assertAlmostEqual(row["rmse"], math.sqrt(5 / 3))
        self.assertEqual(row["model"], "Movie mean")
        self.assertFalse(math.isnan(row["precision@1"]))

    def test_model_without_ranking_gets_nan_ranking(self):
        row = evaluate(GlobalMean(), TRAIN, TEST, k=1)
        self.assertFalse(math.isnan(row["rmse"]))
        self.assertTrue(math.isnan(row["ndcg@1"]))

    def test_model_without_predictions_gets_nan_rmse(self):
        row = evaluate(Popularity(), TRAIN, TEST, k=1)
        self.assertTrue(math.isnan(row["rmse"]))
        self.assertTrue(math.isnan(row["mae"]))
        self.assertFalse(math.isnan(row["precision@1"]))


class TestRunExperiments(unittest.TestCase):
    def test_one_row_per_model_and_split(self):
        ratings = pd.concat([TRAIN, TEST], ignore_index=True)
        ratings["datetime"] = pd.to_datetime(range(len(ratings)), unit="D")
        results = run_experiments([GlobalMean(), MovieMean()], ratings=ratings, k=1)
        self.assertEqual(len(results), 4)
        self.assertEqual(set(results["split"]), {"user_time", "random"})
        self.assertEqual(list(results.columns[:2]), ["split", "model"])


class TestSaveResults(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "results.csv"
        pd.DataFrame(
            {
                "split": ["user_time", "user_time", "random"],
                "model": ["A", "B", "A"],
                "rmse": [1.0, 2.0, 3.0],
            }
        ).to_csv(self.path, index=False)

    def tearDown(self):
        self.dir.cleanup()

    def test_replaces_same_model_and_keeps_others(self):
        new = pd.DataFrame({"split": ["user_time"], "model": ["A"], "rmse": [0.5]})
        table = save_results(new, self.path)
        saved = pd.read_csv(self.path)
        self.assertEqual(len(saved), 3)
        rows = saved.set_index(["split", "model"])["rmse"]
        self.assertEqual(rows[("user_time", "A")], 0.5)  # replaced
        self.assertEqual(rows[("user_time", "B")], 2.0)  # untouched
        self.assertEqual(rows[("random", "A")], 3.0)  # same model, other split: kept
        self.assertEqual(list(table["model"]), ["A", "B", "A"])  # order unchanged

    def test_adds_new_model_at_the_end(self):
        new = pd.DataFrame({"split": ["random"], "model": ["C"], "rmse": [9.0]})
        saved = save_results(new, self.path)
        self.assertEqual(len(saved), 4)
        self.assertEqual(saved.iloc[-1]["model"], "C")

    def test_creates_file_if_missing(self):
        self.path.unlink()
        save_results(pd.DataFrame({"split": ["random"], "model": ["C"]}), self.path)
        self.assertTrue(self.path.exists())


if __name__ == "__main__":
    unittest.main()
