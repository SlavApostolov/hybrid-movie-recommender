# Hybrid Movie Recommender

A movie recommender system that combines **collaborative filtering** and **content-based filtering**, and measures whether the hybrid beats each approach on its own.

Course project for *Applying Artificial Intelligence*, topic 4: "Recommender systems - hybrid approach".

## How it works

| Approach | Idea | Methods |
|---|---|---|
| Collaborative filtering | "Users who rated like you also liked X" | User/item KNN, SVD |
| Content-based | "You liked *Alien*, so here are movies with similar genres, tags and plot" | TF-IDF, cosine similarity |
| Hybrid | Blends both scores; helps most for movies with few ratings (cold start) | Weighted blend |

All models are evaluated on the same train/test split with RMSE, MAE, Precision@K, Recall@K and NDCG@K.

## Dataset

[MovieLens `ml-latest-small`](https://grouplens.org/datasets/movielens/): 100,836 ratings from 610 users on 9,724 movies (1996-2018). The data is not stored in the repo; the download script fetches it.

## Setup

Requires **Python 3.13**.

```bash
git clone https://github.com/SlavApostolov/hybrid-movie-recommender.git
cd hybrid-movie-recommender
git checkout dev
```

Create and activate a virtual environment:

```bash
# Windows (Git Bash)
py -3.13 -m venv .venv
source .venv/Scripts/activate

# macOS / Linux
python3.13 -m venv .venv
source .venv/bin/activate
```

Install the dependencies and download the data:

```bash
python -m pip install -r requirements.txt
python src/download_data.py
```

> On macOS/Linux, remove the `pywinpty` line from `requirements.txt` first (it is Windows-only).

## Usage

```bash
jupyter lab                 # open the notebooks in notebooks/
streamlit run app/app.py    # demo app (coming soon)
```

## Results

*Coming soon.* The comparison of all models will be in [`docs/results.csv`](docs/results.csv).

| Model | RMSE | MAE | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|---|---|
| Baselines | | | | | |
| KNN | | | | | |
| SVD | | | | | |
| Content-based | | | | | |
| Hybrid | | | | | |

## Project structure

```
app/         Streamlit demo
data/        MovieLens files (git-ignored, created by the download script)
docs/        Documentation, project overview, results
models/      Saved trained models (git-ignored)
notebooks/   Exploration and experiments
src/         Reusable code (data loading, splits, models, metrics)
```

See [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md) for the roadmap and team responsibilities.

## Team

- **Slav Apostolov**: data, collaborative filtering
- **Aram**: content-based model
- **Hristo**: evaluation, hybrid model, demo app

## License and citation

Code under the [MIT License](LICENSE).

MovieLens data: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1-19:19. https://doi.org/10.1145/2827872
