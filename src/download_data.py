"""Download and extract the MovieLens dataset into data/raw (skips if present)."""
from pathlib import Path
import zipfile
import requests

DATASET = "ml-latest-small"
URL = f"https://files.grouplens.org/datasets/movielens/{DATASET}.zip"

# It's uppercase cause this are constants that will not be touch in the future 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / f"{DATASET}.zip"
EXTRACTED_DIR = RAW_DIR / DATASET


def download_movielens():
    if (EXTRACTED_DIR / "ratings.csv").exists():
        print("MovieLens is already downloaded, skipping.")
        return
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Folder ready: {RAW_DIR}")

    response = requests.get(URL, timeout=30)
    response.raise_for_status()
    print(f"Download {len(response.content) / 1_000_000:.1f} MB")

    ZIP_PATH.write_bytes(response.content)
    print(f"Saved zip to {ZIP_PATH}")

    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(RAW_DIR)
    ZIP_PATH.unlink()
    print(f"Extracted to {EXTRACTED_DIR}")

if __name__ == "__main__":
    download_movielens()