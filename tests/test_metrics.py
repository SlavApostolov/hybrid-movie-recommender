"""Tests for src/metrics.py. Run with: python -m unittest discover tests"""
import math
import unittest

import pandas as pd

from src.metrics import (
    evaluate_ranking,
    mae,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    relevant_items,
    rmse,
)


class TestRatingMetrics(unittest.TestCase):
    def test_perfect_prediction_is_zero(self):
        self.assertEqual(rmse([4, 3, 5], [4, 3, 5]), 0.0)
        self.assertEqual(mae([4, 3, 5], [4, 3, 5]), 0.0)

    def test_known_values(self):
        # errors: 1, -1, 2  ->  MAE = 4/3,  RMSE = sqrt((1 + 1 + 4) / 3) = sqrt(2)
        self.assertAlmostEqual(mae([4, 3, 5], [3, 4, 3]), 4 / 3)
        self.assertAlmostEqual(rmse([4, 3, 5], [3, 4, 3]), math.sqrt(2))

    def test_rmse_is_at_least_mae(self):
        self.assertGreaterEqual(rmse([1, 5, 2], [2, 2, 2]), mae([1, 5, 2], [2, 2, 2]))


class TestRankingMetrics(unittest.TestCase):
    recommended = [10, 20, 30, 40, 50]
    relevant = {20, 50, 99}

    def test_precision(self):
        self.assertAlmostEqual(precision_at_k(self.recommended, self.relevant, 5), 2 / 5)
        self.assertAlmostEqual(precision_at_k(self.recommended, self.relevant, 2), 1 / 2)

    def test_precision_penalises_short_lists(self):
        self.assertAlmostEqual(precision_at_k([20], self.relevant, 5), 1 / 5)

    def test_recall(self):
        self.assertAlmostEqual(recall_at_k(self.recommended, self.relevant, 5), 2 / 3)
        self.assertAlmostEqual(recall_at_k(self.recommended, self.relevant, 1), 0.0)

    def test_recall_and_ndcg_undefined_without_relevant(self):
        self.assertTrue(math.isnan(recall_at_k(self.recommended, set(), 5)))
        self.assertTrue(math.isnan(ndcg_at_k(self.recommended, set(), 5)))

    def test_ndcg_known_value(self):
        # hits at positions 2 and 5 -> DCG = 1/log2(3) + 1/log2(6)
        # 3 relevant, k=5 -> IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4)
        dcg = 1 / math.log2(3) + 1 / math.log2(6)
        idcg = 1 + 1 / math.log2(3) + 1 / 2
        self.assertAlmostEqual(ndcg_at_k(self.recommended, self.relevant, 5), dcg / idcg)

    def test_ndcg_perfect_and_order_matters(self):
        self.assertAlmostEqual(ndcg_at_k([1, 2, 3], {1, 2}, 3), 1.0)
        early = ndcg_at_k([1, 9, 8], {1}, 3)
        late = ndcg_at_k([9, 8, 1], {1}, 3)
        self.assertGreater(early, late)


class TestEvaluateRanking(unittest.TestCase):
    test = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 3],
            "movieId": [10, 20, 30, 10, 40, 50],
            "rating": [5.0, 4.0, 2.0, 4.5, 3.0, 1.0],
        }
    )

    def test_relevant_items_uses_threshold(self):
        # user 3 liked nothing, so they are left out
        self.assertEqual(relevant_items(self.test), {1: {10, 20}, 2: {10}})

    def test_averages_over_users(self):
        recs = {1: [10, 30], 2: [40, 10]}
        result = evaluate_ranking(recs, self.test, k=2)
        # user 1: precision 1/2, recall 1/2;  user 2: precision 1/2, recall 1
        self.assertAlmostEqual(result["precision@2"], 0.5)
        self.assertAlmostEqual(result["recall@2"], 0.75)
        self.assertEqual(result["n_users"], 2)

    def test_missing_user_scores_zero(self):
        result = evaluate_ranking({1: [10, 20]}, self.test, k=2)
        self.assertAlmostEqual(result["recall@2"], 0.5)  # (1 + 0) / 2


if __name__ == "__main__":
    unittest.main()
