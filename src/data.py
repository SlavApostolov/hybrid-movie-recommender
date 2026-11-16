"""Load the MovieLens ratings and split them into train and test sets.

Every model in the project (baselines, KNN, SVD, content-based, hybrid) should
use these functions, so that all results are computed on the same data.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.download_data import EXTRACTED_DIR

RANDOM_STATE = 42


def load_ratings():
    """Return the ratings table with an extra `datetime` column."""
    ratings = pd.read_csv(EXTRACTED_DIR / "ratings.csv")
    ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")
    return ratings


def random_split(ratings, test_size=0.2, random_state=RANDOM_STATE):
    """Shuffle the ratings and hold out `test_size` of them as the test set."""
    train, test = train_test_split(
        ratings, test_size=test_size, random_state=random_state
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def time_split(ratings, test_size=0.2):
    """Train on the oldest ratings, test on the newest `test_size` share.

    The cutoff is a date: everything before it is train, everything from it on
    is test. This mimics real use (predicting the future from the past).
    Ratings made at the exact same second stay together, so the test share is
    close to, but not exactly, `test_size`.
    """
    cutoff = ratings["datetime"].quantile(1 - test_size)
    train = ratings[ratings["datetime"] < cutoff]
    test = ratings[ratings["datetime"] >= cutoff]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def user_time_split(ratings, test_size=0.2):
    """For every user, hold out their newest `test_size` share of ratings.

    Unlike `time_split`, every user keeps older ratings in train, so the model
    knows each test user. This is the split we use for the final experiments.
    """
    ordered = ratings.sort_values(["userId", "datetime"])
    position = ordered.groupby("userId").cumcount()
    n_ratings = ordered.groupby("userId")["userId"].transform("size")
    is_test = position >= n_ratings * (1 - test_size)
    return (
        ordered[~is_test].reset_index(drop=True),
        ordered[is_test].reset_index(drop=True),
    )
