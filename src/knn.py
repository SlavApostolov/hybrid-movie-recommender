"""User-based k-nearest-neighbours collaborative filtering.

Idea: to guess how user U would rate movie M, find the users whose taste is
most similar to U (the "neighbours") and look at how they rated M.

Everything happens in `fit`:

1. Build the ratings matrix: one row per user, one column per movie.
2. Mean-centre it: subtract each user's average rating. A generous user's 4 and
   a harsh user's 4 are not the same opinion, so we work with how much a rating
   is above or below that user's own average (the "deviation").
3. Similarity of two users = cosine of their deviation rows. Only positive
   similarities are kept (users who disagree are ignored, not "inverted").
4. Each user keeps their `n_neighbors` most similar users.
5. Score for (U, M) = U's average + the neighbours' average deviation on M,
   weighted by similarity. `damping` shrinks the deviation toward 0 when only
   a little similarity weight has rated M, so one enthusiastic neighbour
   cannot push a movie to the top of the list.

`predict` and `recommend` then only look up the finished score matrix. The
class has the same interface as the baselines (`fit` / `predict` / `recommend`
and `name`), so the evaluation code treats all models alike.

Limits: the matrix is dense (users x movies), fine for ml-latest-small, too
big for ml-25m. A user who is not in the training data gets no recommendations.
"""
import numpy as np

MIN_RATING, MAX_RATING = 0.5, 5.0


class UserKNN:
    def __init__(self, n_neighbors=50, damping=1.0):
        self.n_neighbors = n_neighbors
        self.damping = damping
        self.name = f"User KNN (k={n_neighbors}, damping={damping})"

    def fit(self, train):
        self.global_mean = train["rating"].mean()
        self.user_ids = np.sort(train["userId"].unique())
        self.movie_ids = np.sort(train["movieId"].unique())
        self._user_row = {user: row for row, user in enumerate(self.user_ids)}
        self._movie_col = {movie: col for col, movie in enumerate(self.movie_ids)}

        # 1. ratings matrix, plus a True/False matrix of which cells are rated
        rows = train["userId"].map(self._user_row).to_numpy()
        cols = train["movieId"].map(self._movie_col).to_numpy()
        ratings = np.zeros((len(self.user_ids), len(self.movie_ids)), dtype=np.float32)
        rated = np.zeros(ratings.shape, dtype=bool)
        ratings[rows, cols] = train["rating"].to_numpy()
        rated[rows, cols] = True

        # 2. mean-centre: deviation from the user's own average (0 where unrated)
        self.user_means = ratings.sum(axis=1) / rated.sum(axis=1)
        deviations = np.where(rated, ratings - self.user_means[:, None], 0.0)

        # 3. cosine similarity between users, positive values only
        norms = np.linalg.norm(deviations, axis=1)
        norms[norms == 0] = 1.0  # a user who always gives the same rating has no direction
        similarity = (deviations @ deviations.T) / np.outer(norms, norms)
        np.fill_diagonal(similarity, 0.0)  # a user is not their own neighbour
        similarity = np.clip(similarity, 0.0, None)

        # 4. keep only each user's n_neighbors most similar users
        k = min(self.n_neighbors, len(self.user_ids) - 1)
        top = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
        row_index = np.arange(len(similarity))[:, None]
        weights = np.zeros_like(similarity)
        weights[row_index, top] = similarity[row_index, top]

        # 5. weighted average deviation of the neighbours who rated each movie
        numerator = weights @ deviations
        denominator = weights @ rated.astype(np.float32) + self.damping
        deviation = np.divide(
            numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
        )
        self.scores = self.user_means[:, None] + deviation  # not clipped: keeps ranking detail
        self.seen = rated
        return self

    def predict(self, df):
        """Predicted rating (clipped to 0.5-5) for each (userId, movieId) row of `df`.

        Unknown user -> global mean. Known user, movie never seen in train -> user's mean.
        """
        user_idx = df["userId"].map(self._user_row).to_numpy(dtype=float)  # NaN if unknown
        movie_idx = df["movieId"].map(self._movie_col).to_numpy(dtype=float)
        user_known = ~np.isnan(user_idx)
        both_known = user_known & ~np.isnan(movie_idx)

        predictions = np.full(len(df), self.global_mean, dtype=float)
        predictions[user_known] = self.user_means[user_idx[user_known].astype(int)]
        predictions[both_known] = self.scores[
            user_idx[both_known].astype(int), movie_idx[both_known].astype(int)
        ]
        return np.clip(predictions, MIN_RATING, MAX_RATING)

    def recommend(self, user_id, k=10):
        """Top-k movieIds with the highest score among the movies the user has not rated."""
        if user_id not in self._user_row:
            return []
        row = self._user_row[user_id]
        scores = np.where(self.seen[row], -np.inf, self.scores[row])
        best = np.argsort(-scores, kind="stable")[:k]  # stable: ties keep a fixed order
        best = best[np.isfinite(scores[best])]
        return self.movie_ids[best].tolist()
