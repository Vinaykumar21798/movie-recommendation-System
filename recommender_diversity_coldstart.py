"""Task 7: All-user MMR diversity and cold-start evaluation."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer

from config import MMR_LAMBDAS, OUTPUT_DIR, TOP_K, ensure_project_dirs, set_reproducible_seed
from recommender_common import (
    build_user_movie_matrix,
    build_user_movie_sets,
    create_mappings,
    evaluate_recommender,
    load_movielens,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    temporal_train_validation_test_split,
    verify_temporal_order,
    write_csv,
    write_json,
)


def parse_genres(genre_string: str) -> list[str]:
    """Split a MovieLens genre string into clean genre labels."""
    if pd.isna(genre_string) or not genre_string:
        return []
    return [genre for genre in str(genre_string).split("|") if genre and genre != "(no genres listed)"]


def normalize_scores(scores: dict[int, float]) -> dict[int, float]:
    """Min-max normalize a score dictionary to [0, 1]."""
    if not scores:
        return {}
    values = list(scores.values())
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return {key: 1.0 for key in scores}
    return {key: (value - lo) / (hi - lo) for key, value in scores.items()}


class MMRReRanker:
    """Maximal Marginal Relevance re-ranker."""

    def __init__(self, lambda_param: float) -> None:
        self.lambda_param = lambda_param

    def rerank(
        self,
        candidate_movies: list[int],
        relevance_scores: dict[int, float],
        normalized_feature_matrix: np.ndarray,
        movie_idx_map: dict[int, int],
        top_n: int = TOP_K,
    ) -> list[int]:
        """Re-rank movieIds by lambda*relevance - (1-lambda)*max_similarity."""
        selected: list[int] = []
        remaining = set(candidate_movies)
        norm_relevance = normalize_scores(relevance_scores)

        while remaining and len(selected) < top_n:
            best_movie = None
            best_score = -np.inf
            for movie_id in remaining:
                relevance = norm_relevance[movie_id]
                if not selected:
                    diversity_penalty = 0.0
                else:
                    movie_vec = normalized_feature_matrix[movie_idx_map[movie_id]]
                    selected_indices = [movie_idx_map[selected_id] for selected_id in selected]
                    diversity_penalty = float(
                        np.max(normalized_feature_matrix[selected_indices].dot(movie_vec))
                    )
                mmr_score = self.lambda_param * relevance - (1.0 - self.lambda_param) * diversity_penalty
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_movie = movie_id
            if best_movie is None:
                break
            selected.append(best_movie)
            remaining.remove(best_movie)
        return selected


class GenreColdStart:
    """Content-based cold-start helper using genre vectors."""

    def __init__(self, movies_df: pd.DataFrame) -> None:
        self.movies_df = movies_df.copy()
        self.movies_df["genre_list"] = self.movies_df["genres"].apply(parse_genres)
        self.mlb = MultiLabelBinarizer()
        self.genre_matrix = self.mlb.fit_transform(self.movies_df["genre_list"]).astype(float)
        self.all_genres = list(self.mlb.classes_)
        self.movie_idx = {int(movie_id): idx for idx, movie_id in enumerate(self.movies_df["movieId"])}

        norms = np.linalg.norm(self.genre_matrix, axis=1)
        norms[norms == 0] = 1.0
        self.normalized_genre_matrix = self.genre_matrix / norms[:, None]

    def encode_genres(self, genre_list: list[str]) -> np.ndarray:
        """Encode a new movie's genres into the fitted genre space."""
        row = np.zeros((1, len(self.all_genres)))
        genre_to_col = {genre: idx for idx, genre in enumerate(self.all_genres)}
        for genre in genre_list:
            if genre in genre_to_col:
                row[0, genre_to_col[genre]] = 1.0
        return row

    def recommend_for_new_user(
        self,
        liked_titles: list[str],
        top_n: int = TOP_K,
        popularity: dict[int, int] | None = None,
    ) -> list[tuple[str, float]]:
        """Recommend catalog movies from liked titles only."""
        liked_movies = self.movies_df[self.movies_df["title"].isin(liked_titles)]
        if liked_movies.empty:
            return []

        liked_genres: set[str] = set()
        for genres in liked_movies["genres"]:
            liked_genres.update(parse_genres(str(genres)))

        liked_set = set(liked_titles)
        results: list[tuple[str, int, float, int]] = []
        for row in self.movies_df.itertuples(index=False):
            title = str(row.title)
            if title in liked_set:
                continue
            movie_genres = set(parse_genres(str(row.genres)))
            if not movie_genres:
                continue
            overlap = len(liked_genres & movie_genres)
            if overlap == 0:
                continue
            jaccard = overlap / len(liked_genres | movie_genres)
            pop = popularity.get(int(row.movieId), 0) if popularity else 0
            results.append((title, overlap, jaccard, pop))
        results.sort(key=lambda item: (-item[1], -item[2], -item[3], item[0]))
        return [(title, float(jaccard)) for title, _, jaccard, _ in results[:top_n]]

    def recommend_for_new_movie(
        self,
        genre_list: list[str],
        title: str | None = None,
        top_n: int = TOP_K,
        popularity: dict[int, int] | None = None,
    ) -> list[tuple[str, float]]:
        """Find similar catalog movies for a new item with no interactions."""
        new_genres = set(genre_list)
        results: list[tuple[str, int, float, int]] = []
        for row in self.movies_df.itertuples(index=False):
            movie_title = str(row.title)
            if title is not None and movie_title == title:
                continue
            movie_genres = set(parse_genres(str(row.genres)))
            if not movie_genres:
                continue
            overlap = len(new_genres & movie_genres)
            if overlap == 0:
                continue
            jaccard = overlap / len(new_genres | movie_genres)
            pop = popularity.get(int(row.movieId), 0) if popularity else 0
            results.append((movie_title, overlap, jaccard, pop))
        results.sort(key=lambda item: (-item[1], -item[2], -item[3], item[0]))
        return [(title_out, float(jaccard)) for title_out, _, jaccard, _ in results[:top_n]]


def diversity_score(movie_ids: list[int], normalized_feature_matrix: np.ndarray, movie_idx_map: dict[int, int]) -> float:
    """Average pairwise dissimilarity among recommended movieIds."""
    valid_ids = [movie_id for movie_id in movie_ids if movie_id in movie_idx_map]
    if len(valid_ids) < 2:
        return 0.0
    diversities: list[float] = []
    for i in range(len(valid_ids)):
        vec_i = normalized_feature_matrix[movie_idx_map[valid_ids[i]]]
        for j in range(i + 1, len(valid_ids)):
            vec_j = normalized_feature_matrix[movie_idx_map[valid_ids[j]]]
            diversities.append(1.0 - float(vec_i.dot(vec_j)))
    return float(np.mean(diversities)) if diversities else 0.0


def build_popularity_rank(train_df: pd.DataFrame) -> dict[int, int]:
    """Return movieId -> popularity rank, where rank 1 is most popular."""
    counts = train_df["movieId"].value_counts()
    return {int(movie_id): rank for rank, movie_id in enumerate(counts.index, start=1)}


def novelty_score(movie_ids: list[int], popularity_rank: dict[int, int], catalog_size: int) -> float:
    """Average normalized popularity rank; higher means less globally popular."""
    if not movie_ids or catalog_size <= 1:
        return 0.0
    default_rank = catalog_size
    values = [
        (popularity_rank.get(int(movie_id), default_rank) - 1) / (catalog_size - 1)
        for movie_id in movie_ids
    ]
    return float(np.mean(values))


def fit_mf_scores(train_matrix: sp.csr_matrix, n_factors: int = 50) -> np.ndarray:
    """Fit SVD and return dense prediction scores."""
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    user_factors = svd.fit_transform(train_matrix)
    movie_factors = svd.components_.T
    return user_factors.dot(movie_factors.T)


def top_candidate_pool(
    scores: np.ndarray,
    watched: set[int],
    idx_to_movie: dict[int, int],
    movie_idx_map: dict[int, int],
    pool_size: int = 50,
) -> tuple[list[int], dict[int, float]]:
    """Return candidate movieIds and relevance scores from dense MF scores."""
    masked = scores.copy()
    if watched:
        masked[list(watched)] = -np.inf
    top_indices = [int(idx) for idx in np.argsort(masked)[::-1] if masked[idx] > -np.inf]

    movie_ids: list[int] = []
    relevance: dict[int, float] = {}
    for movie_idx in top_indices:
        movie_id = idx_to_movie[movie_idx]
        if movie_id not in movie_idx_map:
            continue
        movie_ids.append(movie_id)
        relevance[movie_id] = float(masked[movie_idx])
        if len(movie_ids) == pool_size:
            break
    return movie_ids, relevance


def evaluate_mmr_strategy(
    pred_scores: np.ndarray,
    train_user_watched: dict[int, set[int]],
    eval_user_movies: dict[int, set[int]],
    idx_to_movie: dict[int, int],
    movie_to_idx: dict[int, int],
    cold_start: GenreColdStart,
    popularity_rank: dict[int, int],
    lambda_param: float | None,
    num_users: int,
    num_movies: int,
    pool_size: int = 50,
    k: int = TOP_K,
) -> dict[str, float]:
    """Evaluate pure MF or MMR-reranked MF for every user."""
    reranker = MMRReRanker(lambda_param) if lambda_param is not None else None
    p_list: list[float] = []
    r_list: list[float] = []
    n_list: list[float] = []
    d_list: list[float] = []
    novelty_list: list[float] = []
    all_recs: set[int] = set()

    for user_idx in range(num_users):
        watched = train_user_watched.get(user_idx, set())
        candidates, relevance = top_candidate_pool(
            pred_scores[user_idx],
            watched,
            idx_to_movie,
            cold_start.movie_idx,
            pool_size=pool_size,
        )
        if reranker is None:
            rec_movie_ids = candidates[:k]
        else:
            rec_movie_ids = reranker.rerank(
                candidates,
                relevance,
                cold_start.normalized_genre_matrix,
                cold_start.movie_idx,
                top_n=k,
            )
        rec_indices = [movie_to_idx[movie_id] for movie_id in rec_movie_ids if movie_id in movie_to_idx]
        all_recs.update(rec_indices)
        test_movies = eval_user_movies.get(user_idx, set())
        p_list.append(precision_at_k(rec_indices, test_movies, k))
        r_list.append(recall_at_k(rec_indices, test_movies, k))
        n_list.append(ndcg_at_k(rec_indices, test_movies, k))
        d_list.append(diversity_score(rec_movie_ids, cold_start.normalized_genre_matrix, cold_start.movie_idx))
        novelty_list.append(novelty_score(rec_movie_ids, popularity_rank, num_movies))

    ndcg = float(np.mean(n_list))
    diversity = float(np.mean(d_list))
    novelty = float(np.mean(novelty_list))
    return {
        "precision@10": float(np.mean(p_list)),
        "recall@10": float(np.mean(r_list)),
        "ndcg@10": ndcg,
        "coverage": len(all_recs) / num_movies if num_movies else 0.0,
        "diversity": diversity,
        "novelty": novelty,
        "utility": ndcg + diversity,
    }


def main() -> None:
    """Run full all-user MMR and cold-start validation."""
    ensure_project_dirs()
    set_reproducible_seed()

    print("=" * 72)
    print("TASK 7: DIVERSITY (MMR) AND COLD-START RECOMMENDATIONS")
    print("=" * 72)

    ratings, movies = load_movielens()
    user_to_idx, _, movie_to_idx, idx_to_movie = create_mappings(ratings, movies)
    num_users = len(user_to_idx)
    num_movies = len(movie_to_idx)

    train_df, val_df, test_df = temporal_train_validation_test_split(ratings)
    verify_temporal_order(train_df, val_df, label="validation")
    verify_temporal_order(pd.concat([train_df, val_df], ignore_index=True), test_df, label="test")

    train_matrix = build_user_movie_matrix(train_df, user_to_idx, movie_to_idx, shape=(num_users, num_movies))
    train_user_watched = build_user_movie_sets(train_df, user_to_idx, movie_to_idx)
    val_user_movies = build_user_movie_sets(val_df, user_to_idx, movie_to_idx)
    test_user_movies = build_user_movie_sets(test_df, user_to_idx, movie_to_idx)

    pred_scores = fit_mf_scores(train_matrix, n_factors=50)
    popularity = train_df["movieId"].value_counts().to_dict()
    popularity_rank = build_popularity_rank(train_df)
    cold_start = GenreColdStart(movies)

    validation_rows: list[dict[str, float]] = []
    baseline_validation = evaluate_mmr_strategy(
        pred_scores,
        train_user_watched,
        val_user_movies,
        idx_to_movie,
        movie_to_idx,
        cold_start,
        popularity_rank,
        lambda_param=None,
        num_users=num_users,
        num_movies=num_movies,
    )

    print("\nValidation lambda sweep:")
    print(f"{'lambda':>8} {'precision@10':>12} {'recall@10':>10} {'ndcg@10':>9} {'coverage':>9} {'diversity':>10} {'novelty':>9} {'utility':>9}")
    for lam in MMR_LAMBDAS:
        metrics = evaluate_mmr_strategy(
            pred_scores,
            train_user_watched,
            val_user_movies,
            idx_to_movie,
            movie_to_idx,
            cold_start,
            popularity_rank,
            lambda_param=lam,
            num_users=num_users,
            num_movies=num_movies,
        )
        row = {"lambda": float(lam), **metrics}
        validation_rows.append(row)
        print(
            f"{lam:8.1f} {metrics['precision@10']:12.4f} {metrics['recall@10']:10.4f} "
            f"{metrics['ndcg@10']:9.4f} {metrics['coverage']:9.4f} "
            f"{metrics['diversity']:10.4f} {metrics['novelty']:9.4f} {metrics['utility']:9.4f}"
        )

    best_row = max(validation_rows, key=lambda row: row["utility"])
    best_lambda = float(best_row["lambda"])
    print(f"\nSelected lambda from validation utility=NDCG@10+Diversity: {best_lambda:.1f}")

    baseline_test = evaluate_mmr_strategy(
        pred_scores,
        train_user_watched,
        test_user_movies,
        idx_to_movie,
        movie_to_idx,
        cold_start,
        popularity_rank,
        lambda_param=None,
        num_users=num_users,
        num_movies=num_movies,
    )
    mmr_test = evaluate_mmr_strategy(
        pred_scores,
        train_user_watched,
        test_user_movies,
        idx_to_movie,
        movie_to_idx,
        cold_start,
        popularity_rank,
        lambda_param=best_lambda,
        num_users=num_users,
        num_movies=num_movies,
    )

    comparison_rows = [
        {"method": "Before MMR", "lambda": None, **baseline_test},
        {"method": "After MMR", "lambda": best_lambda, **mmr_test},
    ]
    print("\nFinal test-set before/after comparison:")
    print(f"{'Method':<14} {'Precision@10':>12} {'Recall@10':>10} {'NDCG@10':>9} {'Coverage':>9} {'Diversity':>10} {'Novelty':>9} {'Utility':>9}")
    print("-" * 98)
    for row in comparison_rows:
        print(
            f"{row['method']:<14} {row['precision@10']:12.4f} {row['recall@10']:10.4f} "
            f"{row['ndcg@10']:9.4f} {row['coverage']:9.4f} "
            f"{row['diversity']:10.4f} {row['novelty']:9.4f} {row['utility']:9.4f}"
        )

    liked_titles = ["Toy Story (1995)", "Shrek (2001)"]
    cold_user_recs = cold_start.recommend_for_new_user(liked_titles, top_n=10, popularity=popularity)
    new_movie_title = "Space Pals (2026)"
    new_movie_genres = ["Animation", "Adventure", "Comedy", "Fantasy", "Children"]
    new_movie_vec = cold_start.encode_genres(new_movie_genres)
    cold_movie_recs = cold_start.recommend_for_new_movie(
        new_movie_genres,
        title=new_movie_title,
        top_n=10,
        popularity=popularity,
    )
    cold_start_validation = {
        "new_user_non_empty": bool(cold_user_recs),
        "new_user_excludes_liked_titles": all(title not in liked_titles for title, _ in cold_user_recs),
        "new_movie_vector_has_genres": bool(new_movie_vec.sum() > 0),
        "new_movie_non_empty": bool(cold_movie_recs),
    }

    print("\nCold-start validation:")
    for key, value in cold_start_validation.items():
        print(f"  {key}: {'PASS' if value else 'FAIL'}")
    print(f"  new-user sample: {[title for title, _ in cold_user_recs[:5]]}")
    print(f"  new-movie sample: {[title for title, _ in cold_movie_recs[:5]]}")

    write_csv(OUTPUT_DIR / "mmr_lambda_sweep.csv", validation_rows)
    write_json(
        OUTPUT_DIR / "diversity_mmr_results.json",
        {
            "validation_baseline": baseline_validation,
            "validation_sweep": validation_rows,
            "selected_lambda": best_lambda,
            "selection_rule": "max(validation NDCG@10 + validation Diversity)",
            "test_comparison": comparison_rows,
            "cold_start_validation": cold_start_validation,
            "cold_start_user_sample": cold_user_recs[:10],
            "cold_start_movie_sample": cold_movie_recs[:10],
        },
    )


if __name__ == "__main__":
    main()
