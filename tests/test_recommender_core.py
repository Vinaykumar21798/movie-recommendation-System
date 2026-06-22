"""Unit tests for shared recommender correctness guarantees."""

from __future__ import annotations

import pandas as pd

from recommender_common import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    temporal_train_validation_test_split,
    verify_temporal_order,
)
from recommender_diversity_coldstart import GenreColdStart
from recommender_graph import cluster_quality


def test_temporal_train_validation_test_split_has_no_leakage() -> None:
    rows = []
    for user_id in [1, 2]:
        for idx in range(10):
            rows.append(
                {
                    "userId": user_id,
                    "movieId": 100 + idx,
                    "rating": 4.0,
                    "timestamp": idx,
                }
            )
    ratings = pd.DataFrame(rows)
    train, validation, test = temporal_train_validation_test_split(ratings)

    verify_temporal_order(train, validation, label="validation")
    verify_temporal_order(pd.concat([train, validation], ignore_index=True), test, label="test")

    for user_id in [1, 2]:
        assert train[train.userId == user_id].timestamp.max() < validation[validation.userId == user_id].timestamp.min()
        assert validation[validation.userId == user_id].timestamp.max() < test[test.userId == user_id].timestamp.min()


def test_ranking_metrics_are_binary_and_position_aware() -> None:
    recs = [10, 20, 30, 40]
    truth = {20, 40, 99}

    assert precision_at_k(recs, truth, 4) == 0.5
    assert recall_at_k(recs, truth, 4) == 2 / 3
    assert 0.0 < ndcg_at_k(recs, truth, 4) < 1.0


def test_cluster_quality_penalizes_giant_component() -> None:
    genre_labels = {0: "Action", 1: "Action", 2: "Drama", 3: "Comedy"}
    giant = cluster_quality([[0, 1, 2, 3]], genre_labels, num_movies=4)
    smaller = cluster_quality([[0, 1], [2], [3]], genre_labels, num_movies=4)

    assert giant["largest_component_ratio"] == 1.0
    assert smaller["purity"] == 1.0
    assert smaller["selection_score"] > giant["selection_score"]


def test_genre_cold_start_returns_valid_recommendations() -> None:
    movies = pd.DataFrame(
        [
            {"movieId": 1, "title": "Toy Story (1995)", "genres": "Animation|Children|Comedy"},
            {"movieId": 2, "title": "Space Comedy", "genres": "Animation|Comedy"},
            {"movieId": 3, "title": "Legal Drama", "genres": "Drama"},
        ]
    )
    cold = GenreColdStart(movies)

    user_recs = cold.recommend_for_new_user(["Toy Story (1995)"], top_n=2)
    movie_recs = cold.recommend_for_new_movie(["Animation", "Comedy"], top_n=2)

    assert user_recs
    assert all(title != "Toy Story (1995)" for title, _ in user_recs)
    assert movie_recs[0][0] in {"Toy Story (1995)", "Space Comedy"}

