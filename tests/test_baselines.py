"""Tests for src/baselines.py. Run with: python -m unittest discover tests"""
import unittest

import pandas as pd

from src.baselines import GlobalMean, MovieMean, Popularity

TRAIN = pd.DataFrame(
    {
        "userId": [1, 1, 2, 2, 3, 3],
        "movieId": [10, 20, 10, 30, 10, 40],
        "rating": [4.0, 2.0, 5.0, 5.0, 3.0, 5.0],
    }
)  # global mean = 24 / 6 = 4.0


class TestGlobalMean(unittest.TestCase):
    def test_predicts_global_mean(self):
        model = GlobalMean().fit(TRAIN)
        self.assertEqual(list(model.predict(TRAIN.head(2))), [4.0, 4.0])
        self.assertEqual(model.recommend(1), [])


class TestMovieMean(unittest.TestCase):
    def test_predicts_movie_mean_and_falls_back(self):
        model = MovieMean().fit(TRAIN)
        test = pd.DataFrame({"userId": [1, 1], "movieId": [10, 999]})
        # movie 10: (4 + 5 + 3) / 3 = 4.0; movie 999 is unknown -> global mean
        self.assertEqual(list(model.predict(test)), [4.0, 4.0])

    def test_damping_pulls_toward_global_mean(self):
        # movie 30 has one 5.0 rating: (5 + 2 * 4.0) / (1 + 2) = 13 / 3
        model = MovieMean(damping=2).fit(TRAIN)
        test = pd.DataFrame({"userId": [1], "movieId": [30]})
        self.assertAlmostEqual(model.predict(test)[0], 13 / 3)

    def test_recommend_skips_seen_movies(self):
        model = MovieMean().fit(TRAIN)
        # ranking: 30 and 40 (5.0), 10 (4.0), 20 (2.0); user 2 has seen 10 and 30
        self.assertEqual(model.recommend(2, k=5), [40, 20])


class TestPopularity(unittest.TestCase):
    def test_recommends_most_rated_unseen(self):
        model = Popularity().fit(TRAIN)
        self.assertEqual(model.recommend(1, k=1)[0] in {30, 40}, True)
        self.assertNotIn(10, model.recommend(1, k=3))  # most popular, but seen

    def test_unknown_user_gets_most_popular(self):
        self.assertEqual(Popularity().fit(TRAIN).recommend(999, k=1), [10])


if __name__ == "__main__":
    unittest.main()
