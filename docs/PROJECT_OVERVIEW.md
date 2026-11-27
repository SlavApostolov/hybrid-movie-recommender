# Hybrid Movie Recommender: Project Overview

*Course: Applying Artificial Intelligence (AAI), topic 4: "Recommender systems - hybrid approach".*
*Status as of 27 Nov 2026. Update the checkboxes as work is merged into `dev`.*

## 1. What we are building

A movie recommender that combines two approaches and measures whether the combination beats each one alone:

- **Collaborative filtering (CF):** "users who rated like you also liked X". Learns from the user x movie ratings matrix (KNN and SVD).
- **Content-based:** "you liked Alien, so here are movies with similar genres, tags and plot". Learns from movie descriptions (TF-IDF and cosine similarity).
- **Hybrid:** blends the two scores. It should help most on movies with few ratings (the cold-start problem).

**Dataset:** MovieLens `ml-latest-small` (100,836 ratings, 610 users, 9,724 movies). Optional stretch: `ml-1m` or `ml-25m`.

**Course deliverables:** (1) documentation: 1.1 Introduction, 1.2 Theory, 1.3 Example, 1.4 Experiments; (2) code; (3) data.

**Final product:** a public GitHub repo, a results table comparing all models, and a Streamlit demo (pick a user, see recommendations).

## 2. Team and responsibilities

| Person | Role | Owns |
|---|---|---|
| **Slav** (repo owner) | Foundation and Collaborative Filtering | Repo and data setup, EDA, train/test split, baselines (with Hristo), user/item KNN, SVD, CF theory, section 1.3 |
| **Aram** | Content-Based Model | Genre/tag/plot features, TF-IDF, similarity, TMDB enrichment, cold-start tests, content theory |
| **Hristo** | Evaluation, Hybrid and App | Metrics code, evaluation pipeline, hybrid blending, Streamlit demo, README, sections 1.1 and 1.4, final assembly |

Everyone: make real commits, review at least one teammate's Pull Request, and be able to explain **all** models at the defense.

## 3. Roadmap and status

Legend: `[x]` done and merged into `dev`, `[ ]` not done yet.

### Week 0: Setup

- [x] **Slav:** GitHub repo, `main` and `dev` branches, folder structure, `.gitignore`, `requirements.txt`
- [x] **Slav:** virtual environment (Python 3.13), data download script `src/download_data.py` (PR #2)
- [x] **Slav:** teammates added to the repo
- [x] **Hristo:** clone, create venv, `pip install -r requirements.txt`, run `python src/download_data.py`
- [ ] **Aram:** clone, create venv, `pip install -r requirements.txt`, run `python src/download_data.py`
- [x] **Hristo:** README with overview, setup steps and results (skeleton done; full version in Week 5)

### Week 1: Foundations and baselines

- [x] **Slav:** EDA notebook `notebooks/01_explore_ratings.ipynb`: rating distribution, sparsity, ratings per movie and per user (PR #3)
- [x] **Slav:** train/test split in `src/data.py`: random, global time-based, per-user time-based (PR #4)
- [x] **Hristo:** metrics: RMSE, MAE, Precision@K, Recall@K, NDCG@K in `src/metrics.py`, with tests (PR #6)
- [x] **Slav + Hristo:** baselines: global mean, per-movie mean, popularity ranking in `src/baselines.py`, results in `notebooks/02_baselines.ipynb` and `docs/results.csv` (done by Hristo)
- [x] **Aram:** explore `movies.csv` and `tags.csv`, clean genres, plan the feature set (done by Slav: `notebooks/03_explore_movies.ipynb`, PR #9)
- [ ] **Hristo:** documentation 1.1 Introduction (idea, terminology, approaches)

**Done when:** a baseline results table exists and the metrics code is tested.

### Week 2: Collaborative filtering and first content features

- [x] **Slav:** user-based KNN by hand in `src/knn.py`, tuned in `notebooks/04_knn.ipynb` (PR #10, #11)
- [ ] **Slav:** item-based KNN; compare with `surprise`
- [ ] **Slav:** matrix factorization (SVD) with hyperparameter tuning
- [ ] **Aram:** TF-IDF matrix of genres and tags, similarity function
- [x] **Hristo:** evaluation pipeline that runs any model on the same split and saves results to `docs/results.csv` (`src/evaluation.py`; rebuild the table with `python -m src.evaluation`)
- [ ] **Slav:** draft CF theory text

**Done when:** KNN and SVD beat the baselines in the results table.

### Week 3: Content-based

- [ ] **Aram:** content-based recommender (item profile, user profile, cosine similarity)
- [ ] **Aram:** enrich with plot summaries via the TMDB API (key in `.env`, never committed)
- [ ] **Aram:** cold-start experiment on movies with few or no ratings
- [ ] **Slav:** improve CF (tuning, compare split types)
- [ ] **Hristo:** Streamlit app skeleton (user picker, top-N list)
- [ ] **Aram:** draft content-based theory text

**Done when:** the content-based model works and cold-start results are measured.

### Week 4: Hybrid and experiments

- [ ] **Hristo:** hybrid v1 (weighted blend of CF and content scores); optional v2 (switch to content for movies with few ratings)
- [ ] **Slav + Aram:** tune blend weights; ablation (CF only vs content only vs hybrid)
- [ ] **All:** final experiments, tables and plots
- [ ] **Hristo:** documentation 1.4 Experiments
- [ ] **Slav + Aram:** finalize theory section 1.2

**Done when:** the final comparison table exists and the hybrid is analysed honestly, even if it does not win everywhere.

### Week 5: Demo, polish, delivery

- [ ] **Hristo:** finish the Streamlit demo (user history, recommendations with posters, "why")
- [ ] **Slav:** clean the code, docstrings, a few unit tests
- [ ] **Aram:** make notebooks reproducible (run top to bottom), `data/README.md`
- [ ] **All:** each person writes the README section for their own part
- [ ] **All:** assemble and proofread documentation 1.1-1.4, cite MovieLens (Harper and Konstan, 2015)
- [ ] **All:** rehearse the defense; every person explains every model
- [ ] Repo description and topics; optional deploy on Streamlit Community Cloud

### Stretch goals (only if ahead of schedule)

- Compare on `ml-1m` or `ml-25m`
- LLM-generated explanations ("recommended because you liked X")
- Neural collaborative filtering (small PyTorch model)

## 4. Schedule note

Week 1 took about two weeks, so the project is roughly one week behind the original 5-week plan. Aram's and Hristo's Week 1 and Week 2 tasks do not depend on Slav's work, so they can start **in parallel now**. As of 27 Nov, Hristo's metrics, baselines and README skeleton are merged; Aram has no commits yet. Suggested catch-up:

- Hristo starts the metrics code and Aram starts the movies/tags exploration immediately.
- Run the content-based work (Aram) alongside CF (Slav) in Weeks 2-3.
- Protect the experiments and the documentation first, and trim the polish of the final week if needed.

## 5. Findings so far

| Topic | Result | Why it matters |
|---|---|---|
| Size | 100,836 ratings, 610 users, 9,724 movies | Small enough for a laptop |
| Sparsity | 98.3% of user x movie cells are empty (density 1.7%) | Models must guess tastes from little data |
| Rating scale | 0.5 to 5.0; mean about 3.5; most common is 4.0 | Ratings skew positive |
| Ratings per movie | median 3, mean 10.4, max 329; 3,446 movies (35%) have exactly 1 rating | Long tail: little information for most movies. This motivates the content-based half of the hybrid |
| Top movies | The 100 most-rated movies hold 16% of all ratings | Popularity is a strong baseline |
| Ratings per user | min 20, median 70.5, mean 165.3, max 2,698 | Every user has enough history, so user cold-start does not appear in this dataset |
| Time range | 29 Mar 1996 to 24 Sep 2018 | Time-based splits are meaningful |

### Train/test split results (80/20)

| Split | Train / Test | Test movies seen in train | Test users seen in train |
|---|---|---|---|
| Random (`random_split`) | 80,668 / 20,168 | 96.0% | (all) |
| Global time (`time_split`) | 80,668 / 20,168 | 84.9% | **9.5%** |
| Per-user time (`user_time_split`) | 80,896 / 19,940 | 91.6% | 100% |

**Decision:** the global time split leaves most test users unseen in training, because most users rate everything in one short burst. We use `user_time_split` for the final experiments and `random_split` as an easier comparison. Everyone must use the functions in `src/data.py`, so that results stay comparable.

## 6. Team rules

**Git workflow**
- `main` holds only stable milestones. `dev` is the shared integration branch.
- Each task goes on its own branch, for example `feature/<short-name>`, created from `dev`.
- When done: push the branch with `git push -u origin <branch>` and open a Pull Request into `dev`. A teammate reviews it before it is merged.
- Start every task with `git checkout dev` then `git pull`.
- Merge `dev` into `main` at milestones (end of a week).

**Shared conventions**
- Ratings tables are DataFrames with columns `userId`, `movieId`, `rating`, `timestamp`, `datetime`.
- Always load and split data through `src/data.py`. Reusable code goes in `src/`; notebooks only explore and present results.
- Every model's scores go into one shared table, `docs/results.csv`. Score models with `src/evaluation.py` and save with `save_results`, which replaces only your own rows. Add finished models to `default_models()` there.
- Secrets (the TMDB key) live in `.env`, which is git-ignored. Never commit them.
- Before committing a notebook: **Kernel, Restart and Run All Cells**.

## 7. Setup for a teammate

```bash
git clone https://github.com/SlavApostolov/hybrid-movie-recommender.git
cd hybrid-movie-recommender
git checkout dev
py -3.13 -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
python src/download_data.py
```

Notes (Windows and Git Bash):
- Activate the venv with `source .venv/Scripts/activate` (PowerShell: `.venv\Scripts\activate`).
- Use `python -m pip ...` rather than a bare `pip`.
- If `python` still runs the wrong interpreter after activating, check `type -a python` for an alias that overrides it.
- Python 3.13 is recommended. `scikit-surprise` (Week 2) is much more likely to install on it than on 3.14.

## 8. Repository map

| Path | Contents |
|---|---|
| `src/download_data.py` | Downloads and extracts MovieLens into `data/raw/` |
| `src/data.py` | `load_ratings`, `random_split`, `time_split`, `user_time_split` |
| `src/metrics.py` | `rmse`, `mae`, `precision_at_k`, `recall_at_k`, `ndcg_at_k`, `evaluate_ranking` |
| `src/evaluation.py` | `evaluate`, `run_experiments`, `save_results` (updates only your own rows in `docs/results.csv`); `python -m src.evaluation` rebuilds the whole table |
| `src/knn.py` | `UserKNN` (user-based collaborative filtering) |
| `tests/` | Unit tests: `python -m unittest discover tests` |
| `notebooks/01_explore_ratings.ipynb` | EDA of the ratings |
| `notebooks/02_baselines.ipynb` | Baseline results table |
| `src/baselines.py` | `GlobalMean`, `MovieMean`, `Popularity` (shared `fit` / `predict` / `recommend` interface) |
| `docs/` | Documentation (1.1-1.4), this overview, `results.csv` |
| `models/` | Saved trained models (git-ignored) |
| `app/` | Streamlit demo |
| `data/` | MovieLens files (git-ignored, downloaded by the script) |
