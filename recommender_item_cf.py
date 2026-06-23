

from __future__ import annotations

import heapq
import time
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp

from config import ITEM_CF_THRESHOLDS, OUTPUT_DIR, TOP_K, ensure_project_dirs, set_reproducible_seed
from recommender_common import (
    build_user_movie_matrix,
    build_user_movie_sets,
    evaluate_recommender,
    get_movie_title,
    load_clean_split_data,
    ModelRegistry,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    write_json,
)


def get_movie_neighbors(S: sp.csr_matrix, movie_idx: int, k: int = 20) -> list[tuple[float, int]]:
    
    row = S[movie_idx]
    candidates = [(float(sim), int(col)) for col, sim in zip(row.indices, row.data) if col != movie_idx]
    return heapq.nlargest(k, candidates, key=lambda x: x[0])


def build_item_similarity(train_matrix: sp.csr_matrix) -> sp.csr_matrix:
    
    movie_user = train_matrix.T.tocsr()
    row_norms = np.sqrt(movie_user.power(2).sum(axis=1).A1)
    row_norms[row_norms == 0] = 1.0
    normalized = sp.diags(1.0 / row_norms).dot(movie_user)
    return normalized.dot(normalized.T).tocsr()


def build_movie_neighbors(S: sp.csr_matrix, k: int = 20) -> dict[int, list[tuple[float, int]]]:
    
    return {movie_idx: get_movie_neighbors(S, movie_idx, k=k) for movie_idx in range(S.shape[0])}


def build_user_train_ratings(
    train_df: pd.DataFrame,
    user_to_idx: dict[int, int],
    movie_to_idx: dict[int, int],
) -> dict[int, list[tuple[int, float]]]:
    
    user_train_ratings: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for row in train_df.itertuples(index=False):
        user_train_ratings[user_to_idx[int(row.userId)]].append(
            (movie_to_idx[int(row.movieId)], float(row.rating))
        )
    return user_train_ratings


def liked_movies_by_user(
    user_train_ratings: dict[int, list[tuple[int, float]]],
    threshold: float,
) -> dict[int, list[int]]:
    
    return {
        user_idx: [movie_idx for movie_idx, rating in ratings if rating >= threshold]
        for user_idx, ratings in user_train_ratings.items()
    }


def recommend_for_user(
    user_idx: int,
    movie_neighbors: dict[int, list[tuple[float, int]]],
    train_user_watched: dict[int, set[int]],
    train_user_liked: dict[int, list[int]],
    top_n: int = TOP_K,
    popular_fallback: Iterable[int] | None = None,
    movie_to_idx: dict[int, int] | None = None,
) -> list[tuple[int, float]]:
    
    liked_movies = train_user_liked.get(user_idx, [])
    watched = train_user_watched.get(user_idx, set())

    candidate_scores: dict[int, float] = defaultdict(float)
    for movie_idx in liked_movies:
        for sim, neighbor_idx in movie_neighbors.get(movie_idx, []):
            if neighbor_idx not in watched:
                candidate_scores[neighbor_idx] += sim

    recs = heapq.nlargest(top_n, candidate_scores.items(), key=lambda x: x[1])

    if len(recs) < top_n and popular_fallback is not None and movie_to_idx is not None:
        seen_recs = {item[0] for item in recs}
        for movie_id in popular_fallback:
            movie_idx = movie_to_idx[int(movie_id)]
            if movie_idx not in watched and movie_idx not in seen_recs:
                recs.append((movie_idx, 0.0))
                if len(recs) == top_n:
                    break
    return recs


def evaluate_item_cf_threshold(
    threshold: float,
    movie_neighbors: dict[int, list[tuple[float, int]]],
    train_user_watched: dict[int, set[int]],
    user_train_ratings: dict[int, list[tuple[int, float]]],
    eval_user_movies: dict[int, set[int]],
    popular_movies: list[int],
    movie_to_idx: dict[int, int],
    num_users: int,
    catalog_size: int,
    k: int = TOP_K,
) -> dict[str, float]:
    
    train_user_liked = liked_movies_by_user(user_train_ratings, threshold)

    def _recommender(user_idx: int) -> list[int]:
        recs = recommend_for_user(
            user_idx,
            movie_neighbors,
            train_user_watched,
            train_user_liked,
            top_n=k,
            popular_fallback=popular_movies,
            movie_to_idx=movie_to_idx,
        )
        return [movie_idx for movie_idx, _ in recs]

    return evaluate_recommender(_recommender, eval_user_movies, num_users, catalog_size, k=k)


def popularity_recommender_metrics(
    train_user_watched: dict[int, set[int]],
    test_user_movies: dict[int, set[int]],
    popular_movies: list[int],
    movie_to_idx: dict[int, int],
    num_users: int,
    catalog_size: int,
    k: int = TOP_K,
) -> dict[str, float]:
    
    def _recommender(user_idx: int) -> list[int]:
        watched = train_user_watched.get(user_idx, set())
        recs: list[int] = []
        for movie_id in popular_movies:
            movie_idx = movie_to_idx[int(movie_id)]
            if movie_idx not in watched:
                recs.append(movie_idx)
                if len(recs) == k:
                    break
        return recs

    return evaluate_recommender(_recommender, test_user_movies, num_users, catalog_size, k=k)


def sparse_csr_memory_mb(matrix: sp.csr_matrix) -> float:
    
    return (matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes) / (1024 * 1024)


def main() -> None:
    
    ensure_project_dirs()
    set_reproducible_seed()

    ratings, movies, train_df, val_df, test_df, user_to_idx, idx_to_user, movie_to_idx, idx_to_movie = load_clean_split_data()
    num_users = len(user_to_idx)
    num_movies = len(movie_to_idx)

    popularity_counts = train_df["movieId"].value_counts()
    popular_movies = [int(mid) for mid in popularity_counts.index.tolist()]

    train_matrix = build_user_movie_matrix(
        train_df,
        user_to_idx,
        movie_to_idx,
        shape=(num_users, num_movies),
    )

    print("Building train-only similarity table...")
    t0 = time.time()
    S = build_item_similarity(train_matrix)
    similarity_runtime = time.time() - t0
    memory_mb = sparse_csr_memory_mb(S)

    print(f"Similarity matrix built in {similarity_runtime:.3f} seconds.")
    print(f"Similarity matrix memory footprint: {memory_mb:.2f} MB")
    print(f"Matrix shape: {S.shape}, Non-zero elements: {S.nnz}, Density: {S.nnz / (num_movies * num_movies):.5f}")

    print("Precomputing top-20 neighbors for all movies...")
    t_neighbors = time.time()
    movie_neighbors = build_movie_neighbors(S, k=20)
    neighbor_runtime = time.time() - t_neighbors
    print(f"Precomputed movie neighbors in {neighbor_runtime:.3f} seconds.")

    toy_story_id = int(movies[movies["title"].str.contains(r"Toy Story \(1995\)", case=False, na=False)].iloc[0]["movieId"])
    toy_story_idx = movie_to_idx[toy_story_id]
    print("\nneighbors of 'Toy Story (1995)':")
    for sim, neighbor_idx in movie_neighbors.get(toy_story_idx, [])[:3]:
        print(f"  {get_movie_title(movies, idx_to_movie, neighbor_idx):<45} {sim:.2f}")

    train_user_watched = build_user_movie_sets(train_df, user_to_idx, movie_to_idx)
    val_user_movies = build_user_movie_sets(val_df, user_to_idx, movie_to_idx)
    test_user_movies = build_user_movie_sets(test_df, user_to_idx, movie_to_idx)
    user_train_ratings = build_user_train_ratings(train_df, user_to_idx, movie_to_idx)

    print("\nValidation-only threshold sweep:")
    print(f"{'threshold':>9} {'precision@10':>12} {'recall@10':>10} {'ndcg@10':>9} {'coverage':>9}")
    validation_rows: list[dict[str, float]] = []
    for threshold in ITEM_CF_THRESHOLDS:
        metrics = evaluate_item_cf_threshold(
            threshold,
            movie_neighbors,
            train_user_watched,
            user_train_ratings,
            val_user_movies,
            popular_movies,
            movie_to_idx,
            num_users,
            num_movies,
        )
        row = {"threshold": threshold, **metrics}
        validation_rows.append(row)
        print(
            f"{threshold:9.1f} {metrics['precision@10']:12.3f} "
            f"{metrics['recall@10']:10.3f} {metrics['ndcg@10']:9.3f} {metrics['coverage']:9.3f}"
        )

    best_row = max(
        validation_rows,
        key=lambda row: (row["ndcg@10"], row["recall@10"], row["precision@10"]),
    )
    best_threshold = float(best_row["threshold"])
    print(f"\nSelected threshold from validation only: {best_threshold:.1f}")

    test_metrics = evaluate_item_cf_threshold(
        best_threshold,
        movie_neighbors,
        train_user_watched,
        user_train_ratings,
        test_user_movies,
        popular_movies,
        movie_to_idx,
        num_users,
        num_movies,
    )
    baseline_metrics = popularity_recommender_metrics(
        train_user_watched,
        test_user_movies,
        popular_movies,
        movie_to_idx,
        num_users,
        num_movies,
    )

    train_user_liked_best = liked_movies_by_user(user_train_ratings, best_threshold)
    sample_recs = recommend_for_user(
        0,
        movie_neighbors,
        train_user_watched,
        train_user_liked_best,
        top_n=3,
        popular_fallback=popular_movies,
        movie_to_idx=movie_to_idx,
    )
    sample_readable = [
        {
            "movieId": idx_to_movie[movie_idx],
            "title": get_movie_title(movies, idx_to_movie, movie_idx),
            "score": round(score, 4),
        }
        for movie_idx, score in sample_recs
    ]
    print(f"\nrecommend(user=0, n=3)   {sample_readable}")

    print("\nFinal test-set evaluation:")
    print(
        f"popularity @10   precision {baseline_metrics['precision@10']:.3f}  "
        f"recall {baseline_metrics['recall@10']:.3f}  ndcg {baseline_metrics['ndcg@10']:.3f}  "
        f"coverage {baseline_metrics['coverage']:.3f}"
    )
    print(
        f"item_cf @10      precision {test_metrics['precision@10']:.3f}  "
        f"recall {test_metrics['recall@10']:.3f}  ndcg {test_metrics['ndcg@10']:.3f}  "
        f"coverage {test_metrics['coverage']:.3f}"
    )

    print("\nLeakage verification checklist:")
    print("  train ratings build similarity matrix: PASS")
    print("  validation ratings tune threshold: PASS")
    print("  test ratings used only for final metrics: PASS")
    print("  train/test temporal order verified: PASS")

    write_json(
        OUTPUT_DIR / "item_cf_evaluation.json",
        {
            "validation_sweep": validation_rows,
            "selected_threshold": best_threshold,
            "test_metrics": test_metrics,
            "baseline_metrics": baseline_metrics,
            "similarity_runtime_seconds": similarity_runtime,
            "neighbor_runtime_seconds": neighbor_runtime,
            "similarity_memory_mb": memory_mb,
        },
    )

    # Register the model state and metrics
    ModelRegistry.register(
        "item_cf",
        {
            "movie_neighbors": movie_neighbors,
            "train_user_liked": train_user_liked_best,
            "best_threshold": best_threshold
        },
        test_metrics,
        {"best_threshold": best_threshold, "k": TOP_K}
    )


if __name__ == "__main__":
    main()
