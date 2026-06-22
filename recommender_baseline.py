

from __future__ import annotations

import numpy as np

from config import TOP_K, set_reproducible_seed
from recommender_common import (
    build_user_movie_sets,
    create_mappings,
    download_and_extract_dataset,
    evaluate_recommender,
    load_movielens,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    temporal_split,
    verify_temporal_order,
)


def main() -> None:
    
    set_reproducible_seed()
    ratings, movies = load_movielens()
    user_to_idx, _, movie_to_idx, _ = create_mappings(ratings, movies)
    num_users = len(user_to_idx)
    catalog_size = len(movie_to_idx)

    train_df, test_df = temporal_split(ratings, test_ratio=0.2)
    verify_temporal_order(train_df, test_df, label="test")

    popularity_counts = train_df["movieId"].value_counts()
    popular_movies = [int(mid) for mid in popularity_counts.index.tolist()]
    print(f"popularity top-5   {popular_movies[:5]}")

    train_user_watched = build_user_movie_sets(train_df, user_to_idx, movie_to_idx)
    test_user_movies = build_user_movie_sets(test_df, user_to_idx, movie_to_idx)

    def make_recommender(k: int):
        def _recommender(user_idx: int) -> list[int]:
            watched = train_user_watched.get(user_idx, set())
            recs: list[int] = []
            for movie_id in popular_movies:
                movie_idx = movie_to_idx[movie_id]
                if movie_idx not in watched:
                    recs.append(movie_idx)
                    if len(recs) == k:
                        break
            return recs

        return _recommender

    metrics_10 = evaluate_recommender(make_recommender(TOP_K), test_user_movies, num_users, catalog_size, k=TOP_K)
    print(
        f"baseline @10       precision {metrics_10['precision@10']:.3f}  "
        f"recall {metrics_10['recall@10']:.3f}  ndcg {metrics_10['ndcg@10']:.3f}  "
        f"coverage {metrics_10['coverage']:.3f}"
    )
    print()
    print(f"{'k':>2}   {'precision':<10} {'recall':<10} {'ndcg':<10} {'coverage':<10}")

    for k in range(1, 21):
        metrics = evaluate_recommender(make_recommender(k), test_user_movies, num_users, catalog_size, k=k)
        print(
            f"{k:>2}     {metrics[f'precision@{k}']:.3f}      {metrics[f'recall@{k}']:.3f}      "
            f"{metrics[f'ndcg@{k}']:.3f}      {metrics['coverage']:.3f}"
        )


if __name__ == "__main__":
    main()
