"""Task 5: Matrix factorization with portable artifacts and validation tuning."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

from config import ARTIFACT_DIR, MF_FACTORS, OUTPUT_DIR, TOP_K, ensure_project_dirs, set_reproducible_seed
from recommender_common import (
    build_user_movie_matrix,
    build_user_movie_sets,
    create_mappings,
    evaluate_recommender,
    get_movie_title,
    load_movielens,
    temporal_train_validation_test_split,
    verify_temporal_order,
    write_csv,
    write_json,
)
from recommender_item_cf import (
    build_item_similarity,
    build_movie_neighbors,
    build_user_train_ratings,
    evaluate_item_cf_threshold,
    popularity_recommender_metrics,
)


def fit_svd(train_matrix: sp.csr_matrix, n_factors: int) -> tuple[TruncatedSVD, np.ndarray, np.ndarray, np.ndarray]:
    """Fit TruncatedSVD and return model, user factors, movie factors, and dense scores."""
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    user_factors = svd.fit_transform(train_matrix)
    movie_factors = svd.components_.T
    pred_scores = user_factors.dot(movie_factors.T)
    return svd, user_factors, movie_factors, pred_scores


def evaluate_pred_scores(
    pred_scores: np.ndarray,
    train_user_watched: dict[int, set[int]],
    eval_user_movies: dict[int, set[int]],
    num_users: int,
    num_movies: int,
    k: int = TOP_K,
) -> dict[str, float]:
    """Evaluate dense user-by-movie score matrix with seen-item masking."""
    def _recommender(user_idx: int) -> list[int]:
        scores = pred_scores[user_idx].copy()
        watched = train_user_watched.get(user_idx, set())
        if watched:
            scores[list(watched)] = -np.inf
        return [int(idx) for idx in np.argsort(scores)[::-1][:k] if scores[idx] > -np.inf]

    return evaluate_recommender(_recommender, eval_user_movies, num_users, num_movies, k=k)


def top_recommendations_from_scores(
    scores: np.ndarray,
    watched: set[int],
    top_n: int,
) -> list[int]:
    """Return top-N movie indices after masking watched movies."""
    masked = scores.copy()
    if watched:
        masked[list(watched)] = -np.inf
    return [int(idx) for idx in np.argsort(masked)[::-1][:top_n] if masked[idx] > -np.inf]


def nearest_movies_in_factor_space(
    movies: pd.DataFrame,
    idx_to_movie: dict[int, int],
    movie_to_idx: dict[int, int],
    train_df: pd.DataFrame,
    movie_factors: np.ndarray,
    query_titles: list[str],
    min_ratings: int = 15,
    top_n: int = 4,
) -> dict[str, list[dict[str, object]]]:
    """Find nearest movies in normalized SVD factor space."""
    norms = np.linalg.norm(movie_factors, axis=1)
    norms[norms == 0] = 1.0
    normalized = movie_factors / norms[:, None]
    train_movie_counts = train_df["movieId"].value_counts()
    movie_counts = {movie_to_idx[int(mid)]: int(count) for mid, count in train_movie_counts.items()}

    results: dict[str, list[dict[str, object]]] = {}
    for title_query in query_titles:
        pattern = title_query.replace("(", r"\(").replace(")", r"\)")
        matches = movies[movies["title"].str.contains(pattern, case=False, na=False)]
        if matches.empty:
            results[title_query] = []
            continue

        movie_id = int(matches.iloc[0]["movieId"])
        movie_idx = movie_to_idx[movie_id]
        sims = normalized.dot(normalized[movie_idx])
        candidates = [
            idx
            for idx in range(len(idx_to_movie))
            if idx != movie_idx and movie_counts.get(idx, 0) >= min_ratings
        ]
        candidate_sims = sims[candidates]
        top_sub = np.argsort(candidate_sims)[::-1][:top_n]

        rows: list[dict[str, object]] = []
        for sub_idx in top_sub:
            idx = candidates[int(sub_idx)]
            movie_id_out = idx_to_movie[idx]
            genres = str(movies[movies["movieId"] == movie_id_out]["genres"].iloc[0])
            rows.append(
                {
                    "movieId": int(movie_id_out),
                    "title": get_movie_title(movies, idx_to_movie, idx),
                    "genres": genres,
                    "similarity": float(sims[idx]),
                }
            )
        results[title_query] = rows
    return results


def main() -> None:
    """Run validation-selected matrix factorization and save portable artifacts."""
    parser = argparse.ArgumentParser(description="Matrix factorization recommender")
    parser.add_argument("--artifact_dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    ensure_project_dirs()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_reproducible_seed()

    print("=" * 72)
    print("TASK 5: MATRIX FACTORIZATION RECOMMENDATIONS")
    print("=" * 72)

    ratings, movies = load_movielens()
    user_to_idx, _, movie_to_idx, idx_to_movie = create_mappings(ratings, movies)
    num_users = len(user_to_idx)
    num_movies = len(movie_to_idx)

    train_df, val_df, test_df = temporal_train_validation_test_split(ratings)
    verify_temporal_order(train_df, val_df, label="validation")
    verify_temporal_order(pd.concat([train_df, val_df], ignore_index=True), test_df, label="test")
    print(f"Dataset split: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test ratings.")
    print(f"Catalog size: {num_users} users, {num_movies} movies.")

    train_matrix = build_user_movie_matrix(
        train_df,
        user_to_idx,
        movie_to_idx,
        shape=(num_users, num_movies),
    )
    train_user_watched = build_user_movie_sets(train_df, user_to_idx, movie_to_idx)
    val_user_movies = build_user_movie_sets(val_df, user_to_idx, movie_to_idx)
    test_user_movies = build_user_movie_sets(test_df, user_to_idx, movie_to_idx)

    print("\nValidation factor sweep:")
    print(f"{'factors':>8} {'time_s':>8} {'precision@10':>12} {'recall@10':>10} {'ndcg@10':>9} {'coverage':>9}")
    sweep_rows: list[dict[str, float]] = []
    fitted_by_factor: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for n_factors in MF_FACTORS:
        t0 = time.perf_counter()
        _, user_factors, movie_factors, pred_scores = fit_svd(train_matrix, n_factors)
        elapsed = time.perf_counter() - t0
        metrics = evaluate_pred_scores(
            pred_scores,
            train_user_watched,
            val_user_movies,
            num_users,
            num_movies,
        )
        row = {"factors": float(n_factors), "train_seconds": elapsed, **metrics}
        sweep_rows.append(row)
        fitted_by_factor[n_factors] = (user_factors, movie_factors, pred_scores)
        print(
            f"{n_factors:8d} {elapsed:8.2f} {metrics['precision@10']:12.4f} "
            f"{metrics['recall@10']:10.4f} {metrics['ndcg@10']:9.4f} {metrics['coverage']:9.4f}"
        )

    best_row = max(sweep_rows, key=lambda row: (row["recall@10"], row["ndcg@10"], row["precision@10"]))
    best_factors = int(best_row["factors"])
    print(f"\nSelected factors from validation only: {best_factors}")

    user_factors_best, movie_factors_best, pred_scores_best = fitted_by_factor[best_factors]
    test_metrics = evaluate_pred_scores(
        pred_scores_best,
        train_user_watched,
        test_user_movies,
        num_users,
        num_movies,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        [int(row["factors"]) for row in sweep_rows],
        [row["recall@10"] for row in sweep_rows],
        marker="o",
        linewidth=2,
    )
    plt.title("Validation Recall@10 vs Latent Factors")
    plt.xlabel("Latent factors")
    plt.ylabel("Recall@10")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_path = args.artifact_dir / "mf_recall_sweep.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    query_titles = [
        "Toy Story (1995)",
        "Matrix, The (1999)",
        "Star Wars: Episode IV - A New Hope (1977)",
    ]
    nearest = nearest_movies_in_factor_space(
        movies,
        idx_to_movie,
        movie_to_idx,
        train_df,
        movie_factors_best,
        query_titles,
    )

    print("\nNearest neighbors in selected factor space:")
    for title, rows in nearest.items():
        print(f"nearest to '{title}':")
        for idx, row in enumerate(rows, start=1):
            print(f"  {idx}. {row['title']} ({row['genres']}) [sim: {row['similarity']:.2f}]")

    popularity_counts = train_df["movieId"].value_counts()
    popular_movies = [int(mid) for mid in popularity_counts.index.tolist()]
    baseline_metrics = popularity_recommender_metrics(
        train_user_watched,
        test_user_movies,
        popular_movies,
        movie_to_idx,
        num_users,
        num_movies,
    )

    S = build_item_similarity(train_matrix)
    movie_neighbors = build_movie_neighbors(S, k=20)
    user_train_ratings = build_user_train_ratings(train_df, user_to_idx, movie_to_idx)
    item_cf_metrics = evaluate_item_cf_threshold(
        3.5,
        movie_neighbors,
        train_user_watched,
        user_train_ratings,
        test_user_movies,
        popular_movies,
        movie_to_idx,
        num_users,
        num_movies,
    )

    print("\nFinal test-set comparison:")
    print(f"{'Method':<36} {'Precision@10':>12} {'Recall@10':>10} {'NDCG@10':>9} {'Coverage':>9}")
    print("-" * 82)
    comparison_rows = [
        {"method": "Popularity Baseline", **baseline_metrics},
        {"method": "Item Collaborative Filtering", **item_cf_metrics},
        {f"method": f"Matrix Factorization SVD k={best_factors}", **test_metrics},
    ]
    for row in comparison_rows:
        print(
            f"{row['method']:<36} {row['precision@10']:12.4f} "
            f"{row['recall@10']:10.4f} {row['ndcg@10']:9.4f} {row['coverage']:9.4f}"
        )

    write_csv(args.output_dir / "mf_factor_sweep.csv", sweep_rows)
    write_json(
        args.output_dir / "mf_results.json",
        {
            "selected_factors": best_factors,
            "selection_rule": "highest validation Recall@10, tie-break NDCG@10 then Precision@10",
            "validation_sweep": sweep_rows,
            "test_metrics": test_metrics,
            "comparison": comparison_rows,
            "factor_dimensions": {
                "user_factors": list(user_factors_best.shape),
                "movie_factors": list(movie_factors_best.shape),
            },
            "plot_path": str(plot_path),
        },
    )
    write_json(args.output_dir / "mf_nearest_neighbors.json", nearest)
    print(f"\nSaved plot: {plot_path}")
    print(f"Saved metrics: {args.output_dir / 'mf_results.json'}")
    print(f"Saved nearest neighbors: {args.output_dir / 'mf_nearest_neighbors.json'}")


if __name__ == "__main__":
    main()
