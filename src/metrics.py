"""Metrics for evaluating the recommenders.

Two kinds of questions, two kinds of metrics:

1. Rating prediction: "how close is the predicted rating to the real one?"
   -> `rmse`, `mae`. Lower is better.
2. Top-K ranking: "are the K movies we recommend ones the user actually liked?"
   -> `precision_at_k`, `recall_at_k`, `ndcg_at_k`. Higher is better (max 1.0).

For ranking, a test movie counts as *relevant* for a user if they rated it
at least `RELEVANCE_THRESHOLD` (4.0, since the average rating is about 3.5).
"""
import numpy as np

RELEVANCE_THRESHOLD = 4.0


# ---------------------------------------------------------------------------
# Rating prediction
# ---------------------------------------------------------------------------

def rmse(y_true, y_pred):
    """Root mean squared error. Punishes big mistakes more than small ones."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    """Mean absolute error: on average, how many stars the prediction is off by."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


# ---------------------------------------------------------------------------
# Top-K ranking (for one user)
# ---------------------------------------------------------------------------

def precision_at_k(recommended, relevant, k):
    """Share of the top-K recommendations that are relevant.

    "Of the 10 movies we showed, how many did the user like?"
    Always divides by K, so returning fewer than K movies is penalised.
    """
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / k


def recall_at_k(recommended, relevant, k):
    """Share of the relevant movies that made it into the top-K.

    "Of all the movies the user liked, how many did we find?"
    Returns NaN when the user has no relevant movies (recall is undefined).
    """
    if len(relevant) == 0:
        return float("nan")
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    """Normalised Discounted Cumulative Gain with binary relevance.

    Like precision, but a hit at position 1 is worth more than a hit at
    position 10 (each hit is divided by log2(position + 1)). The result is
    divided by the best possible score, so a perfect ranking gives 1.0.
    Returns NaN when the user has no relevant movies.
    """
    relevant = set(relevant)
    if len(relevant) == 0:
        return float("nan")
    dcg = sum(
        1 / np.log2(position + 2)  # position is 0-based, so +2
        for position, movie in enumerate(recommended[:k])
        if movie in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1 / np.log2(position + 2) for position in range(ideal_hits))
    return float(dcg / idcg)


# ---------------------------------------------------------------------------
# Whole test set
# ---------------------------------------------------------------------------

def relevant_items(test, threshold=RELEVANCE_THRESHOLD):
    """Return {userId: set of movieIds the user rated >= threshold in test}."""
    liked = test[test["rating"] >= threshold]
    return liked.groupby("userId")["movieId"].apply(set).to_dict()


def evaluate_ranking(recommendations, test, k=10, threshold=RELEVANCE_THRESHOLD):
    """Average Precision@K, Recall@K and NDCG@K over all test users.

    `recommendations` is {userId: [movieId, ...]} ordered best first.
    Only users with at least one relevant test movie are scored. A user the
    model gave no recommendations to scores 0, so skipping users is not rewarded.
    """
    relevant = relevant_items(test, threshold)
    scores = {"precision": [], "recall": [], "ndcg": []}
    for user, liked in relevant.items():
        recommended = list(recommendations.get(user, []))
        scores["precision"].append(precision_at_k(recommended, liked, k))
        scores["recall"].append(recall_at_k(recommended, liked, k))
        scores["ndcg"].append(ndcg_at_k(recommended, liked, k))
    return {
        f"{name}@{k}": float(np.mean(values)) for name, values in scores.items()
    } | {"n_users": len(relevant)}
