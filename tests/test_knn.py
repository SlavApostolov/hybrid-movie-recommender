"""Tests for src/knn.py. Run with: python -m unittest discover tests"""
import unittest

import pandas as pd

from src.knn import UserKNN

# Users A=1, B=2, C=3 and movies 1, 2, 3.
#   A: movie1=5, movie2=1                 -> mean 3,      deviations +2, -2
#   B: movie1=5, movie2=1, movie3=5       -> mean 11/3,   deviations +4/3, -8/3, +4/3
#   C: movie1=1, movie2=5                 -> mean 3,      deviations -2, +2
# A and B like the same movies (similarity 0.866), A and C disagree (similarity < 0, dropped).
TRAIN = pd.DataFrame(
    {
        "userId": [1, 1, 2, 2, 2, 3, 3],
        "movieId": [1, 2, 1, 2, 3, 1, 2],
        "rating": [5.0, 1.0, 5.0, 1.0, 5.0, 1.0, 5.0],
    }
)


def ask(model, user, movie):
    return model.predict(pd.DataFrame({"userId": [user], "movieId": [movie]}))[0]


class TestUserKNN(unittest.TestCase):
    def test_known_value(self):
        # A's only neighbour is B, who rated movie 3 at +4/3 above their own mean.
        # With damping 0 the deviation is used in full: 3 + 4/3.
        model = UserKNN(n_neighbors=1, damping=0).fit(TRAIN)
        self.assertAlmostEqual(ask(model, 1, 3), 3 + 4 / 3, places=3)

    def test_damping_shrinks_the_deviation(self):
        # deviation = 0.866 * (4/3) / (0.866 + damping)
        plain = UserKNN(n_neighbors=1, damping=0).fit(TRAIN)
        damped = UserKNN(n_neighbors=1, damping=1).fit(TRAIN)
        self.assertLess(ask(damped, 1, 3), ask(plain, 1, 3))
        self.assertGreater(ask(damped, 1, 3), 3.0)

    def test_user_without_similar_users_gets_their_own_mean(self):
        model = UserKNN(n_neighbors=1, damping=0).fit(TRAIN)
        self.assertAlmostEqual(ask(model, 3, 3), 3.0, places=5)

    def test_fallbacks_for_unknown_ids(self):
        model = UserKNN(n_neighbors=1).fit(TRAIN)
        self.assertAlmostEqual(ask(model, 999, 1), TRAIN["rating"].mean())  # unknown user
        self.assertAlmostEqual(ask(model, 1, 999), 3.0, places=5)  # unknown movie -> user mean

    def test_predictions_stay_between_half_and_five(self):
        model = UserKNN(n_neighbors=2, damping=0).fit(TRAIN)
        predictions = model.predict(TRAIN)
        self.assertTrue(((predictions >= 0.5) & (predictions <= 5.0)).all())

    def test_recommend_skips_rated_movies(self):
        model = UserKNN(n_neighbors=1).fit(TRAIN)
        self.assertEqual(model.recommend(1, k=5), [3])  # A has only rated movies 1 and 2
        self.assertEqual(model.recommend(2, k=5), [])  # B has rated everything
        self.assertEqual(model.recommend(999, k=5), [])  # unknown user


if __name__ == "__main__":
    unittest.main()
