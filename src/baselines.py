"""Baseline recommenders: the simplest models every real model must beat.

All models share one interface, so the evaluation code can treat them alike:

    model = SomeBaseline().fit(train)
    model.predict(test)           # predicted rating for each test row
    model.recommend(user_id, k)   # top-k movieIds the user has not rated yet

- `GlobalMean`: predicts the average rating of all movies for everything.
  It cannot rank movies (all scores are equal), so it only gets RMSE/MAE.
- `MovieMean`: predicts each movie's average rating. With `damping > 0`, movies
  with few ratings are pulled toward the global mean (a Bayesian average), so a
  movie with a single 5.0 rating does not top the list.
- `Popularity`: recommends the most-rated movies. It does not predict ratings,
  so it only gets ranking metrics.
"""
import numpy as np


class Baseline:
    """Shared logic: remember what each user has seen and rank unseen movies."""

    def fit(self, train):
        self.global_mean = train["rating"].mean()
        self.seen = train.groupby("userId")["movieId"].apply(set).to_dict()
        self.ranking = self._rank_movies(train)
        return self

    def _rank_movies(self, train):
        """Return all movieIds ordered best first. Overridden by subclasses."""
        return []

    def predict(self, df):
        """Predicted rating for each (userId, movieId) row of `df`."""
        raise NotImplementedError

    def recommend(self, user_id, k=10):
        """Top-k movies from the global ranking that the user has not rated."""
        seen = self.seen.get(user_id, set())
        top = []
        for movie in self.ranking:
            if movie not in seen:
                top.append(movie)
                if len(top) == k:
                    break
        return top


class GlobalMean(Baseline):
    name = "Global mean"

    def predict(self, df):
        return np.full(len(df), self.global_mean)

    def recommend(self, user_id, k=10):
        return []  # every movie has the same score, so there is no ranking


class MovieMean(Baseline):
    def __init__(self, damping=0):
        self.damping = damping
        self.name = "Movie mean" if damping == 0 else f"Movie mean (damping={damping})"

    def fit(self, train):
        stats = train.groupby("movieId")["rating"].agg(["sum", "count"])
        global_mean = train["rating"].mean()
        # (sum + damping * global_mean) / (count + damping)
        self.movie_means = (stats["sum"] + self.damping * global_mean) / (
            stats["count"] + self.damping
        )
        return super().fit(train)

    def _rank_movies(self, train):
        return self.movie_means.sort_values(ascending=False).index.tolist()

    def predict(self, df):
        # movies never seen in train fall back to the global mean
        return df["movieId"].map(self.movie_means).fillna(self.global_mean).to_numpy()


class Popularity(Baseline):
    name = "Popularity"

    def _rank_movies(self, train):
        return train["movieId"].value_counts().index.tolist()

    def predict(self, df):
        return np.full(len(df), np.nan)  # popularity ranks, it does not predict ratings
