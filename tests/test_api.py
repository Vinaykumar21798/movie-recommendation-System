"""FastAPI integration tests for all recommendation endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

import recommender_api as api


def test_recommend_endpoints_for_all_methods() -> None:
    api.load_all_models()
    with TestClient(api.app) as client:
        for method in ["popularity", "item_cf", "mf", "graph"]:
            response = client.get(f"/recommend/1?n=3&method={method}")
            assert response.status_code == 200
            payload = response.json()
            assert payload["method"] == method
            assert len(payload["items"]) == 3
            assert {"movieId", "title", "score"} <= set(payload["items"][0])


def test_methods_and_compare_endpoints() -> None:
    api.load_all_models()
    with TestClient(api.app) as client:
        methods_response = client.get("/methods")
        assert methods_response.status_code == 200
        assert set(methods_response.json()["methods"]) == {"popularity", "item_cf", "mf", "graph"}

        compare_response = client.get("/compare")
        assert compare_response.status_code == 200
        payload = compare_response.json()
        assert "winner" in payload
        assert "comparison" in payload
        assert "mf" in payload["comparison"]

        health_response = client.get("/health")
        assert health_response.status_code == 200
        health = health_response.json()
        assert health["status"] == "ok"
        assert health["loaded"] is True
        assert health["users"] > 0
        assert health["movies"] > 0

def test_unknown_user_and_method_validation() -> None:
    api.load_all_models()
    with TestClient(api.app) as client:
        bad_method = client.get("/recommend/1?n=3&method=unknown")
        missing_user = client.get("/recommend/999999?n=3&method=mf")
        assert bad_method.status_code == 400
        assert missing_user.status_code == 404


def test_serving_recommendations_exclude_all_known_ratings() -> None:
    api.load_all_models()
    with TestClient(api.app) as client:
        response = client.get("/recommend/1?n=10&method=mf")
        assert response.status_code == 200
        rec_movie_ids = {item["movieId"] for item in response.json()["items"]}

    user_idx = api.state.user_to_idx[1]
    full_seen_movie_ids = {
        api.state.idx_to_movie[movie_idx]
        for movie_idx in api.state.full_user_watched[user_idx]
    }
    assert rec_movie_ids.isdisjoint(full_seen_movie_ids)
