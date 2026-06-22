

from __future__ import annotations

import csv
import json
import os
import zipfile
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp

from config import DATA_DIR, RANDOM_SEED, ensure_project_dirs, set_reproducible_seed


def download_and_extract_dataset(data_dir: str | os.PathLike[str] = DATA_DIR) -> str:
    
    set_reproducible_seed(RANDOM_SEED)
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    zip_path = data_path / "ml-latest-small.zip"
    extracted_path = data_path / "ml-latest-small"

    if not extracted_path.exists():
        print("Dataset not found. Downloading ml-latest-small...")
        url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with zip_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete. Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_path)
        zip_path.unlink(missing_ok=True)
        print("Extraction complete.")
    else:
        print("Dataset already exists.")

    return str(extracted_path)


def load_movielens(data_dir: str | os.PathLike[str] = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    dataset_path = Path(download_and_extract_dataset(data_dir))
    ratings = pd.read_csv(dataset_path / "ratings.csv")
    movies = pd.read_csv(dataset_path / "movies.csv")
    return ratings, movies


def create_mappings(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    include_unrated_movies: bool = True,
) -> tuple[dict[int, int], dict[int, int], dict[int, int], dict[int, int]]:
    
    unique_users = sorted(ratings["userId"].unique())
    if include_unrated_movies:
        movie_ids = set(movies["movieId"].unique()) | set(ratings["movieId"].unique())
    else:
        movie_ids = set(ratings["movieId"].unique())
    unique_movies = sorted(movie_ids)

    user_to_idx = {int(user_id): idx for idx, user_id in enumerate(unique_users)}
    idx_to_user = {idx: int(user_id) for idx, user_id in enumerate(unique_users)}
    movie_to_idx = {int(movie_id): idx for idx, movie_id in enumerate(unique_movies)}
    idx_to_movie = {idx: int(movie_id) for idx, movie_id in enumerate(unique_movies)}
    return user_to_idx, idx_to_user, movie_to_idx, idx_to_movie


def build_user_movie_matrix(
    ratings_df: pd.DataFrame,
    user_to_idx: dict[int, int],
    movie_to_idx: dict[int, int],
    shape: tuple[int, int] | None = None,
    binary: bool = False,
) -> sp.csr_matrix:
    
    rows = ratings_df["userId"].map(user_to_idx).to_numpy()
    cols = ratings_df["movieId"].map(movie_to_idx).to_numpy()
    values = np.ones(len(ratings_df), dtype=np.float32) if binary else ratings_df["rating"].to_numpy()
    matrix_shape = shape or (len(user_to_idx), len(movie_to_idx))
    return sp.csr_matrix((values, (rows, cols)), shape=matrix_shape)


def temporal_split(ratings_df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, group in ratings_df.groupby("userId"):
        sorted_group = group.sort_values("timestamp")
        n = len(sorted_group)
        test_size = max(1, int(n * test_ratio)) if n > 1 else 0
        split_idx = n - test_size
        train_parts.append(sorted_group.iloc[:split_idx])
        test_parts.append(sorted_group.iloc[split_idx:])

    return (
        pd.concat(train_parts).reset_index(drop=True),
        pd.concat(test_parts).reset_index(drop=True),
    )


def temporal_train_validation_test_split(
    ratings_df: pd.DataFrame,
    validation_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, group in ratings_df.groupby("userId"):
        sorted_group = group.sort_values("timestamp")
        n = len(sorted_group)
        if n < 3:
            train_parts.append(sorted_group)
            continue

        test_size = max(1, int(n * test_ratio))
        val_size = max(1, int(n * validation_ratio))
        if test_size + val_size >= n:
            val_size = max(1, n - test_size - 1)

        train_end = n - test_size - val_size
        val_end = n - test_size
        train_parts.append(sorted_group.iloc[:train_end])
        val_parts.append(sorted_group.iloc[train_end:val_end])
        test_parts.append(sorted_group.iloc[val_end:])

    return (
        pd.concat(train_parts).reset_index(drop=True),
        pd.concat(val_parts).reset_index(drop=True),
        pd.concat(test_parts).reset_index(drop=True),
    )


def verify_temporal_order(
    train_df: pd.DataFrame,
    later_df: pd.DataFrame,
    label: str = "validation/test",
) -> None:
    
    for user_id, train_group in train_df.groupby("userId"):
        later_group = later_df[later_df["userId"] == user_id]
        if later_group.empty or train_group.empty:
            continue
        if train_group["timestamp"].max() > later_group["timestamp"].min():
            raise ValueError(f"Temporal leakage detected for user {user_id} before {label}.")


def precision_at_k(recs: Iterable[int], test_movies: set[int], k: int) -> float:
    
    recs_k = list(recs)[:k]
    if k <= 0:
        return 0.0
    return len(set(recs_k) & test_movies) / k


def recall_at_k(recs: Iterable[int], test_movies: set[int], k: int) -> float:
    
    if not test_movies:
        return 0.0
    recs_k = list(recs)[:k]
    return len(set(recs_k) & test_movies) / len(test_movies)


def ndcg_at_k(recs: Iterable[int], test_movies: set[int], k: int) -> float:
    
    recs_k = list(recs)[:k]
    dcg = 0.0
    for idx, movie in enumerate(recs_k):
        if movie in test_movies:
            dcg += 1.0 / np.log2(idx + 2)
    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(min(k, len(test_movies))))
    return dcg / idcg if idcg else 0.0


def build_user_movie_sets(
    ratings_df: pd.DataFrame,
    user_to_idx: dict[int, int],
    movie_to_idx: dict[int, int],
) -> dict[int, set[int]]:
    
    raw = (
        ratings_df.groupby("userId")["movieId"]
        .apply(lambda s: {movie_to_idx[int(m)] for m in s})
        .to_dict()
    )
    return {user_to_idx[int(user_id)]: mids for user_id, mids in raw.items()}


def evaluate_recommender(
    recommender_fn: Callable[[int], list[int]],
    test_user_movies: dict[int, set[int]],
    num_users: int,
    catalog_size: int,
    k: int = 10,
) -> dict[str, float]:
    
    p_list: list[float] = []
    r_list: list[float] = []
    n_list: list[float] = []
    all_recs: set[int] = set()

    for user_idx in range(num_users):
        recs = recommender_fn(user_idx)[:k]
        all_recs.update(recs)
        test_movies = test_user_movies.get(user_idx, set())
        if test_movies:
            p_list.append(precision_at_k(recs, test_movies, k))
            r_list.append(recall_at_k(recs, test_movies, k))
            n_list.append(ndcg_at_k(recs, test_movies, k))

    precision_val = float(np.mean(p_list)) if p_list else 0.0
    recall_val = float(np.mean(r_list)) if r_list else 0.0
    ndcg_val = float(np.mean(n_list)) if n_list else 0.0
    coverage_val = len(all_recs) / catalog_size if catalog_size else 0.0

    return {
        "precision@10": precision_val,
        "recall@10": recall_val,
        "ndcg@10": ndcg_val,
        f"precision@{k}": precision_val,
        f"recall@{k}": recall_val,
        f"ndcg@{k}": ndcg_val,
        "coverage": coverage_val,
    }


def get_movie_title(movies_df: pd.DataFrame, idx_to_movie: dict[int, int], movie_idx: int) -> str:
    
    movie_id = idx_to_movie.get(int(movie_idx))
    if movie_id is None:
        return "Unknown Movie"
    match = movies_df[movies_df["movieId"] == movie_id]
    if match.empty:
        return "Unknown Movie"
    return str(match.iloc[0]["title"])


def write_json(path: str | os.PathLike[str], payload: object) -> None:
    
    ensure_project_dirs()
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_csv(path: str | os.PathLike[str], rows: list[dict[str, object]]) -> None:
    
    ensure_project_dirs()
    if not rows:
        return
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

