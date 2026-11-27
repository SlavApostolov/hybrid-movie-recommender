"""Evaluation pipeline: run any model on the same split and save its scores.

Every model in the project has the same interface (`fit` / `predict` /
`recommend` and a `name`), so one function can score them all:

    from src.evaluation import evaluate, run_experiments, save_results

    row = evaluate(model, train, test)            # one row of the results table
    results = run_experiments([model_a, model_b])  # every model on every split
    save_results(results)                          # update docs/results.csv

`save_results` replaces only the rows of the models it is given (matched on
split + model name) and keeps everyone else's rows, so teammates can add their
models to the shared table without overwriting each other.

A model that cannot predict ratings (its `predict` returns only NaN, like
`Popularity`) gets NaN for RMSE/MAE. A model that cannot rank (`recommend`
returns nothing, like `GlobalMean`) gets NaN for the ranking metrics.

To rebuild the whole table from scratch: `python -m src.evaluation`
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_ratings, random_split, user_time_split
from src.metrics import evaluate_ranking, mae, rmse

K = 10
RESULTS_PATH = Path(__file__).resolve().parent.parent / "docs" / "results.csv"
KEY = ["split", "model"]

# The splits used in the results table. `user_time` is the main one.
SPLITS = {"user_time": user_time_split, "random": random_split}


def columns(k=K):
    """Column order of the results table."""
    return KEY + ["rmse", "mae", f"precision@{k}", f"recall@{k}", f"ndcg@{k}"]


def evaluate(model, train, test, k=K):
    """Fit `model` on `train`, score it on `test`, and return one row of results."""
    model.fit(train)

    predictions = np.asarray(model.predict(test), dtype=float)
    can_predict = not np.isnan(predictions).all()

    users = test["userId"].unique()
    recommendations = {user: model.recommend(user, k) for user in users}
    can_rank = any(len(movies) > 0 for movies in recommendations.values())
    ranking = evaluate_ranking(recommendations, test, k)

    return {
        "model": model.name,
        "rmse": rmse(test["rating"], predictions) if can_predict else np.nan,
        "mae": mae(test["rating"], predictions) if can_predict else np.nan,
        f"precision@{k}": ranking[f"precision@{k}"] if can_rank else np.nan,
        f"recall@{k}": ranking[f"recall@{k}"] if can_rank else np.nan,
        f"ndcg@{k}": ranking[f"ndcg@{k}"] if can_rank else np.nan,
    }


def run_experiments(models, splits=None, ratings=None, k=K):
    """Evaluate every model on every split and return the results table.

    `splits` is a list of names from `SPLITS` (default: all of them).
    `ratings` defaults to the full MovieLens ratings from `load_ratings()`.
    """
    ratings = load_ratings() if ratings is None else ratings
    splits = list(SPLITS) if splits is None else splits
    rows = []
    for split_name in splits:
        train, test = SPLITS[split_name](ratings)
        for model in models:
            rows.append({"split": split_name} | evaluate(model, train, test, k))
    return pd.DataFrame(rows, columns=columns(k))


def save_results(results, path=RESULTS_PATH):
    """Add `results` to the shared table, replacing older rows of the same models.

    Rows are matched on (split, model). A replaced row stays where it was, rows
    of other models are kept as they are, and new models are added at the end,
    so the table keeps a stable order. Returns the full table that was written.
    """
    results = pd.DataFrame(results)
    path = Path(path)
    if path.exists():
        old = pd.read_csv(path)
        new_rows = {tuple(row[KEY]): row for _, row in results.iterrows()}
        rows = [new_rows.pop(tuple(row[KEY]), row) for _, row in old.iterrows()]
        rows += list(new_rows.values())  # models not in the table yet
        table = pd.DataFrame(rows).reset_index(drop=True)
    else:
        table = results.reset_index(drop=True)
    order = [c for c in columns() if c in table.columns]
    table = table[order + [c for c in table.columns if c not in order]]
    table.to_csv(path, index=False)
    return table


def default_models():
    """All finished models with their chosen settings (tuned in the notebooks)."""
    from src.baselines import GlobalMean, MovieMean, Popularity
    from src.knn import UserKNN

    return [
        GlobalMean(),
        MovieMean(),
        MovieMean(damping=10),
        Popularity(),
        UserKNN(n_neighbors=200, damping=1.0),  # best validation RMSE (notebook 04)
        UserKNN(n_neighbors=50, damping=10.0),  # best validation NDCG@10 (notebook 04)
    ]


if __name__ == "__main__":
    results = run_experiments(default_models())
    save_results(results)
    print(results.round(4).to_string(index=False))
    print(f"\nSaved to {RESULTS_PATH}")
