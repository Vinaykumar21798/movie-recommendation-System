

from __future__ import annotations

import argparse
import functools
import pickle
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any

import heapq
import networkx as nx
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sklearn.decomposition import TruncatedSVD

import uvicorn

from config import API_ARTIFACT_PATH, TOP_K, ensure_project_dirs, set_reproducible_seed
from recommender_common import (
    build_user_movie_matrix,
    build_user_movie_sets,
    create_mappings,
    evaluate_recommender,
    get_movie_title,
    load_movielens,
    temporal_split,
    write_json,
)
from recommender_graph import bfs_recommend, build_bipartite_graph
from recommender_item_cf import (
    build_item_similarity,
    build_movie_neighbors,
    build_user_train_ratings,
    liked_movies_by_user,
    popularity_recommender_metrics,
    recommend_for_user,
)


ARTIFACT_VERSION = 3


class RecommendationItem(BaseModel):
    

    movieId: int
    title: str
    score: float


class RecommendationResponse(BaseModel):
    

    user_id: int
    method: str
    items: list[RecommendationItem]
    latency_ms: float


class MethodsResponse(BaseModel):
    

    methods: list[str]


class MetricRow(BaseModel):
    

    precision_at_10: float
    recall_at_10: float
    ndcg_at_10: float
    coverage: float
    ms_per_req: float


class CompareResponse(BaseModel):
    

    comparison: dict[str, MetricRow]
    winner: dict[str, str]


class HealthResponse(BaseModel):
    

    status: str
    loaded: bool
    users: int
    movies: int
    methods: list[str]


class ModelState:
    

    def __init__(self) -> None:
        self.loaded = False
        self.ratings = None
        self.movies = None
        self.train_df = None
        self.test_df = None
        self.user_to_idx: dict[int, int] = {}
        self.idx_to_user: dict[int, int] = {}
        self.movie_to_idx: dict[int, int] = {}
        self.idx_to_movie: dict[int, int] = {}
        self.num_users = 0
        self.num_movies = 0
        self.train_matrix = None
        self.train_user_watched: dict[int, set[int]] = {}
        self.full_user_watched: dict[int, set[int]] = {}
        self.test_user_movies: dict[int, set[int]] = {}
        self.popular_movies: list[int] = []
        self.popularity_counts: dict[int, int] = {}
        self.movie_neighbors: dict[int, list[tuple[float, int]]] = {}
        self.train_user_liked: dict[int, list[int]] = {}
        self.pred_scores = None
        self.graph = None
        self.comparison: dict[str, dict[str, float]] | None = None

    def to_artifact(self) -> dict[str, Any]:
        
        return {
            "artifact_version": ARTIFACT_VERSION,
            "ratings": self.ratings,
            "movies": self.movies,
            "train_df": self.train_df,
            "test_df": self.test_df,
            "user_to_idx": self.user_to_idx,
            "idx_to_user": self.idx_to_user,
            "movie_to_idx": self.movie_to_idx,
            "idx_to_movie": self.idx_to_movie,
            "num_users": self.num_users,
            "num_movies": self.num_movies,
            "train_matrix": self.train_matrix,
            "train_user_watched": self.train_user_watched,
            "full_user_watched": self.full_user_watched,
            "test_user_movies": self.test_user_movies,
            "popular_movies": self.popular_movies,
            "popularity_counts": self.popularity_counts,
            "movie_neighbors": self.movie_neighbors,
            "train_user_liked": dict(self.train_user_liked),
            "pred_scores": self.pred_scores,
            "graph": self.graph,
            "comparison": self.comparison,
        }

    def load_artifact(self, payload: dict[str, Any]) -> None:
        
        for key, value in payload.items():
            if key != "artifact_version":
                setattr(self, key, value)
        self.loaded = True


state = ModelState()


def _persist_state() -> None:
    
    ensure_project_dirs()
    with API_ARTIFACT_PATH.open("wb") as f:
        pickle.dump(state.to_artifact(), f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_persisted_state() -> bool:
    
    if not API_ARTIFACT_PATH.exists():
        return False
    try:
        with API_ARTIFACT_PATH.open("rb") as f:
            payload = pickle.load(f)
        if payload.get("artifact_version") != ARTIFACT_VERSION:
            return False
        state.load_artifact(payload)
        print(f"Loaded API model artifact from {API_ARTIFACT_PATH}")
        return True
    except Exception as exc:
        print(f"Could not load API artifact; rebuilding. Reason: {exc}")
        return False


def load_all_models(force_rebuild: bool = False) -> None:
    
    if state.loaded and not force_rebuild:
        return

    ensure_project_dirs()
    set_reproducible_seed()

    if not force_rebuild and _load_persisted_state():
        return

    print("=" * 72)
    print("TASK 8: RECOMMENDATION API MODEL BUILD")
    print("=" * 72)
    t_start = time.perf_counter()

    state.ratings, state.movies = load_movielens()
    state.user_to_idx, state.idx_to_user, state.movie_to_idx, state.idx_to_movie = create_mappings(
        state.ratings,
        state.movies,
    )
    state.num_users = len(state.user_to_idx)
    state.num_movies = len(state.movie_to_idx)
    state.train_df, state.test_df = temporal_split(state.ratings, test_ratio=0.2)

    state.train_matrix = build_user_movie_matrix(
        state.train_df,
        state.user_to_idx,
        state.movie_to_idx,
        shape=(state.num_users, state.num_movies),
    )
    state.train_user_watched = build_user_movie_sets(
        state.train_df,
        state.user_to_idx,
        state.movie_to_idx,
    )
    state.full_user_watched = build_user_movie_sets(
        state.ratings,
        state.user_to_idx,
        state.movie_to_idx,
    )
    state.test_user_movies = build_user_movie_sets(
        state.test_df,
        state.user_to_idx,
        state.movie_to_idx,
    )

    pop_series = state.train_df["movieId"].value_counts()
    state.popular_movies = [int(mid) for mid in pop_series.index.tolist()]
    state.popularity_counts = {int(mid): int(count) for mid, count in pop_series.to_dict().items()}
    print("  [1/4] Popularity model ready")

    similarity = build_item_similarity(state.train_matrix)
    state.movie_neighbors = build_movie_neighbors(similarity, k=20)
    user_train_ratings = build_user_train_ratings(state.train_df, state.user_to_idx, state.movie_to_idx)
    state.train_user_liked = liked_movies_by_user(user_train_ratings, threshold=3.5)
    print("  [2/4] Item-CF model ready")

    svd = TruncatedSVD(n_components=50, random_state=42)
    user_factors = svd.fit_transform(state.train_matrix)
    movie_factors = svd.components_.T
    state.pred_scores = user_factors.dot(movie_factors.T)
    print(f"  [3/4] MF model ready (user factors {user_factors.shape})")

    state.graph = build_bipartite_graph(state.train_df)
    print(f"  [4/4] Graph model ready ({state.graph.number_of_nodes()} nodes, {state.graph.number_of_edges()} edges)")

    state.loaded = True
    _persist_state()
    print(f"All models built and persisted in {time.perf_counter() - t_start:.2f}s")


def _recommend_popularity(
    user_idx: int,
    n: int = TOP_K,
    watched_sets: dict[int, set[int]] | None = None,
) -> list[tuple[int, float]]:
    watched_source = watched_sets if watched_sets is not None else state.train_user_watched
    watched = watched_source.get(user_idx, set())
    results: list[tuple[int, float]] = []
    for movie_id in state.popular_movies:
        movie_idx = state.movie_to_idx[movie_id]
        if movie_idx not in watched:
            results.append((movie_idx, float(state.popularity_counts[movie_id])))
            if len(results) == n:
                break
    return results


def _recommend_item_cf(
    user_idx: int,
    n: int = TOP_K,
    watched_sets: dict[int, set[int]] | None = None,
) -> list[tuple[int, float]]:
    watched_source = watched_sets if watched_sets is not None else state.train_user_watched
    recs = recommend_for_user(
        user_idx,
        state.movie_neighbors,
        watched_source,
        state.train_user_liked,
        top_n=n,
        popular_fallback=state.popular_movies,
        movie_to_idx=state.movie_to_idx,
    )
    return recs


def _recommend_mf(
    user_idx: int,
    n: int = TOP_K,
    watched_sets: dict[int, set[int]] | None = None,
) -> list[tuple[int, float]]:
    scores = state.pred_scores[user_idx].copy()
    watched_source = watched_sets if watched_sets is not None else state.train_user_watched
    watched = watched_source.get(user_idx, set())
    if watched:
        scores[list(watched)] = -np.inf
    top_idx = np.argsort(scores)[::-1][:n]
    recs = [(int(movie_idx), float(scores[movie_idx])) for movie_idx in top_idx if scores[movie_idx] > -np.inf]
    if len(recs) < n:
        seen = {item[0] for item in recs}
        for movie_id in state.popular_movies:
            movie_idx = state.movie_to_idx[movie_id]
            if movie_idx not in watched and movie_idx not in seen:
                recs.append((movie_idx, 0.0))
                if len(recs) == n:
                    break
    return recs


def _recommend_graph(
    user_idx: int,
    n: int = TOP_K,
    watched_sets: dict[int, set[int]] | None = None,
) -> list[tuple[int, float]]:
    watched_source = watched_sets if watched_sets is not None else state.train_user_watched
    recs = bfs_recommend(
        state.graph,
        user_idx,
        state.idx_to_user,
        state.movie_to_idx,
        watched_source,
        top_n=n,
        popular_fallback=state.popular_movies,
    )
    return [(movie_idx, float(n - rank)) for rank, movie_idx in enumerate(recs)]


METHODS = {
    "popularity": _recommend_popularity,
    "item_cf": _recommend_item_cf,
    "mf": _recommend_mf,
    "graph": _recommend_graph,
}


def _normalize(items: list[tuple[int, float]]) -> list[tuple[int, float]]:
    if not items:
        return []
    scores = [score for _, score in items]
    lo = min(scores)
    hi = max(scores)
    if hi == lo:
        return [(movie_idx, 1.0) for movie_idx, _ in items]
    return [(movie_idx, (score - lo) / (hi - lo)) for movie_idx, score in items]


@functools.lru_cache(maxsize=4096)
def cached_recommend(user_id: int, n: int, method: str) -> tuple[tuple[int, str, float], ...] | None:
    
    if user_id not in state.user_to_idx:
        return None
    user_idx = state.user_to_idx[user_id]
    raw = METHODS[method](user_idx, n, state.full_user_watched)
    normed = _normalize(raw)
    rows = []
    for movie_idx, score in normed:
        movie_id = state.idx_to_movie[movie_idx]
        rows.append((int(movie_id), get_movie_title(state.movies, state.idx_to_movie, movie_idx), round(float(score), 4)))
    return tuple(rows)


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    load_all_models()
    yield


app = FastAPI(
    title="Movie Recommendation API",
    description="Movie recommendation service with cached popularity, item-CF, MF, and graph recommenders.",
    lifespan=lifespan,
)


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend_endpoint(
    user_id: int,
    n: int = Query(TOP_K, ge=1, le=50),
    method: str = Query("mf", description="popularity | item_cf | mf | graph"),
) -> RecommendationResponse:
    
    if not state.loaded:
        raise HTTPException(503, "Models are still loading")
    if method not in METHODS:
        raise HTTPException(400, f"Unknown method '{method}'. Choose from: {list(METHODS.keys())}")

    t0 = time.perf_counter()
    results = cached_recommend(user_id, n, method)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if results is None:
        raise HTTPException(404, f"User {user_id} not found")

    return RecommendationResponse(
        user_id=user_id,
        method=method,
        items=[
            RecommendationItem(movieId=movie_id, title=title, score=score)
            for movie_id, title, score in results
        ],
        latency_ms=round(elapsed_ms, 3),
    )


@app.get("/methods", response_model=MethodsResponse)
def list_methods() -> MethodsResponse:
    
    return MethodsResponse(methods=list(METHODS.keys()))


@app.get("/health", response_model=HealthResponse)
def health_endpoint() -> HealthResponse:
    
    return HealthResponse(
        status="ok" if state.loaded else "loading",
        loaded=state.loaded,
        users=state.num_users,
        movies=state.num_movies,
        methods=list(METHODS.keys()),
    )


@app.get("/compare", response_model=CompareResponse)
def compare_methods() -> CompareResponse:
    
    if not state.loaded:
        raise HTTPException(503, "Models are still loading")
    if state.comparison is None:
        state.comparison = _build_comparison_results()
        _persist_state()
    winner = _pick_winner(state.comparison)
    return CompareResponse(comparison=_schema_comparison(state.comparison), winner=winner)


def _build_comparison_results() -> dict[str, dict[str, float]]:
    
    results: dict[str, dict[str, float]] = {}
    for method_name, func in METHODS.items():
        t0 = time.perf_counter()

        def _recommender(user_idx: int, f=func) -> list[int]:
            return [movie_idx for movie_idx, _ in f(user_idx, TOP_K, state.train_user_watched)]

        metrics = evaluate_recommender(
            _recommender,
            state.test_user_movies,
            state.num_users,
            state.num_movies,
            k=TOP_K,
        )
        total_s = time.perf_counter() - t0
        metrics["ms_per_req"] = (total_s / state.num_users) * 1000
        results[method_name] = metrics
    return results


def _schema_comparison(results: dict[str, dict[str, float]]) -> dict[str, MetricRow]:
    
    return {
        method: MetricRow(
            precision_at_10=round(metrics["precision@10"], 4),
            recall_at_10=round(metrics["recall@10"], 4),
            ndcg_at_10=round(metrics["ndcg@10"], 4),
            coverage=round(metrics["coverage"], 4),
            ms_per_req=round(metrics["ms_per_req"], 3),
        )
        for method, metrics in results.items()
    }


def _pick_winner(results: dict[str, dict[str, float]], speed_limit_ms: float = 10.0) -> dict[str, str]:
    
    fast_methods = [method for method, row in results.items() if row["ms_per_req"] < speed_limit_ms]
    eligible = fast_methods if fast_methods else list(results.keys())
    winner = max(eligible, key=lambda method: results[method]["ndcg@10"])
    rule = f"best ndcg@10 under {speed_limit_ms:.0f}ms" if fast_methods else "best ndcg@10 overall"
    return {"method": winner, "rule": rule}


def demonstrate_caching() -> dict[str, float]:
    
    cached_recommend.cache_clear()
    user_id = state.idx_to_user[0]
    t0 = time.perf_counter()
    cold = cached_recommend(user_id, 10, "mf")
    cold_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    warm = cached_recommend(user_id, 10, "mf")
    warm_ms = (time.perf_counter() - t1) * 1000
    speedup = cold_ms / max(warm_ms, 1e-9)
    return {
        "user_id": float(user_id),
        "cold_request_ms": cold_ms,
        "warm_request_ms": warm_ms,
        "cache_speedup_x": speedup,
        "result_count": float(len(warm or cold or [])),
    }


def run_validation_report() -> dict[str, Any]:
    
    cache = demonstrate_caching()
    comparison = _build_comparison_results()
    state.comparison = comparison
    winner = _pick_winner(comparison)
    report = {
        "cache_benchmark": cache,
        "comparison": comparison,
        "winner": winner,
        "artifact_path": str(API_ARTIFACT_PATH),
    }
    write_json("outputs/api_validation_report.json", report)
    return report


def main() -> None:
    
    parser = argparse.ArgumentParser(description="Task 8 Recommendation API")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--rebuild-artifacts", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_all_models(force_rebuild=args.rebuild_artifacts)

    if args.eval_only:
        report = run_validation_report()
        cache = report["cache_benchmark"]
        print("\nCache benchmark:")
        print(f"  cold_request_ms={cache['cold_request_ms']:.3f}")
        print(f"  warm_request_ms={cache['warm_request_ms']:.6f}")
        print(f"  speedup={cache['cache_speedup_x']:.1f}x")
        print("\nComparison:")
        print(f"{'method':<12} {'precision':>10} {'recall':>10} {'ndcg':>10} {'coverage':>10} {'ms/req':>10}")
        for method, row in report["comparison"].items():
            print(
                f"{method:<12} {row['precision@10']:10.4f} {row['recall@10']:10.4f} "
                f"{row['ndcg@10']:10.4f} {row['coverage']:10.4f} {row['ms_per_req']:10.3f}"
            )
        print(f"\nwinner {report['winner']['method']} ({report['winner']['rule']})")
        return

    print(f"Starting FastAPI server on http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
