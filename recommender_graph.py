

from __future__ import annotations

import heapq
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

from config import GRAPH_SIMILARITY_THRESHOLDS, OUTPUT_DIR, TOP_K, ensure_project_dirs, set_reproducible_seed
from recommender_common import (
    build_user_movie_matrix,
    build_user_movie_sets,
    create_mappings,
    evaluate_recommender,
    get_movie_title,
    load_movielens,
    temporal_split,
    write_csv,
    write_json,
)
from recommender_item_cf import build_item_similarity, popularity_recommender_metrics


class UnionFind:
    

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: int, right: int) -> bool:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return False
        if self.size[root_left] < self.size[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        self.size[root_left] += self.size[root_right]
        return True


def build_bipartite_graph(train_df: pd.DataFrame) -> nx.Graph:
    
    graph = nx.Graph()
    for row in train_df.itertuples(index=False):
        graph.add_edge(f"u{int(row.userId)}", f"m{int(row.movieId)}")
    return graph


def bfs_recommend(
    graph: nx.Graph,
    user_idx: int,
    idx_to_user: dict[int, int],
    movie_to_idx: dict[int, int],
    train_user_watched: dict[int, set[int]],
    top_n: int = TOP_K,
    popular_fallback: Iterable[int] | None = None,
) -> list[int]:
    
    user_node = f"u{idx_to_user[user_idx]}"
    watched = train_user_watched.get(user_idx, set())
    if not graph.has_node(user_node):
        recs: list[int] = []
        if popular_fallback is not None:
            for movie_id in popular_fallback:
                movie_idx = movie_to_idx.get(int(movie_id))
                if movie_idx is not None and movie_idx not in watched:
                    recs.append(movie_idx)
                    if len(recs) == top_n:
                        break
        return recs

    movies_h1 = list(graph.neighbors(user_node))
    users_h2: set[str] = set()
    for movie_node in movies_h1:
        users_h2.update(graph.neighbors(movie_node))

    movie_counts: dict[str, int] = {}
    for other_user in users_h2:
        for movie_node in graph.neighbors(other_user):
            movie_counts[movie_node] = movie_counts.get(movie_node, 0) + 1

    for movie_node in movies_h1:
        movie_counts.pop(movie_node, None)

    candidates: list[tuple[int, int]] = []
    for movie_node, count in movie_counts.items():
        movie_id = int(movie_node[1:])
        movie_idx = movie_to_idx.get(movie_id)
        if movie_idx is not None and movie_idx not in watched:
            candidates.append((count, movie_idx))
    recs = [movie_idx for _, movie_idx in heapq.nlargest(top_n, candidates, key=lambda x: x[0])]
    if len(recs) < top_n and popular_fallback is not None:
        seen_recs = set(recs)
        for movie_id in popular_fallback:
            movie_idx = movie_to_idx.get(int(movie_id))
            if movie_idx is not None and movie_idx not in watched and movie_idx not in seen_recs:
                recs.append(movie_idx)
                if len(recs) == top_n:
                    break
    return recs


def ppr_scores_from_matrix(train_binary: sp.csr_matrix, alpha: float = 0.85, odd_steps: int = 5) -> sp.csr_matrix:
    
    user_degree = np.asarray(train_binary.sum(axis=1)).ravel()
    movie_degree = np.asarray(train_binary.sum(axis=0)).ravel()
    user_degree[user_degree == 0] = 1.0
    movie_degree[movie_degree == 0] = 1.0

    P_um = sp.diags(1.0 / user_degree).dot(train_binary).tocsr()
    P_mu = sp.diags(1.0 / movie_degree).dot(train_binary.T).tocsr()

    user_state = sp.eye(train_binary.shape[0], format="csr")
    scores = sp.csr_matrix((train_binary.shape[0], train_binary.shape[1]), dtype=np.float32)
    for step in range(odd_steps):
        movie_state = user_state.dot(P_um).tocsr()
        path_length = 2 * step + 1
        scores = scores + ((1.0 - alpha) * (alpha ** path_length)) * movie_state
        user_state = movie_state.dot(P_mu).tocsr()
    return scores.tocsr()


def ppr_recommend_from_scores(
    ppr_scores: sp.csr_matrix,
    user_idx: int,
    train_user_watched: dict[int, set[int]],
    top_n: int = TOP_K,
    popular_fallback: Iterable[int] | None = None,
    movie_to_idx: dict[int, int] | None = None,
) -> list[int]:
    
    scores = ppr_scores[user_idx].toarray().ravel()
    watched = train_user_watched.get(user_idx, set())
    if watched:
        scores[list(watched)] = -np.inf
    recs = [int(idx) for idx in np.argsort(scores)[::-1][:top_n] if scores[idx] > -np.inf]
    if len(recs) < top_n and popular_fallback is not None and movie_to_idx is not None:
        seen_recs = set(recs)
        for movie_id in popular_fallback:
            movie_idx = movie_to_idx.get(int(movie_id))
            if movie_idx is not None and movie_idx not in watched and movie_idx not in seen_recs:
                recs.append(movie_idx)
                if len(recs) == top_n:
                    break
    return recs


def primary_genre_by_movie(movies: pd.DataFrame, movie_to_idx: dict[int, int]) -> dict[int, str]:
    
    labels: dict[int, str] = {}
    for row in movies.itertuples(index=False):
        movie_id = int(row.movieId)
        if movie_id not in movie_to_idx:
            continue
        genres = str(row.genres).split("|")
        labels[movie_to_idx[movie_id]] = genres[0] if genres and genres[0] != "(no genres listed)" else "unknown"
    return labels


def clusters_from_similarity(S: sp.csr_matrix, threshold: float, num_movies: int) -> list[list[int]]:
    
    coo = S.tocoo()
    mask = (coo.row < coo.col) & (coo.data >= threshold)
    uf = UnionFind(num_movies)
    for left, right in zip(coo.row[mask], coo.col[mask]):
        uf.union(int(left), int(right))

    groups: dict[int, list[int]] = defaultdict(list)
    for movie_idx in range(num_movies):
        groups[uf.find(movie_idx)].append(movie_idx)
    return list(groups.values())


def cluster_quality(
    clusters: list[list[int]],
    genre_labels: dict[int, str],
    num_movies: int,
) -> dict[str, float]:
    
    non_singleton = [cluster for cluster in clusters if len(cluster) > 1]
    clustered_movies = sum(len(cluster) for cluster in non_singleton)
    largest_cluster = max((len(cluster) for cluster in clusters), default=0)

    if clustered_movies == 0:
        purity = 0.0
        entropy = 0.0
    else:
        pure_count = 0
        total_entropy = 0.0
        for cluster in non_singleton:
            counts = Counter(genre_labels.get(movie_idx, "unknown") for movie_idx in cluster)
            pure_count += counts.most_common(1)[0][1]
            c_entropy = 0.0
            sz = len(cluster)
            for cnt in counts.values():
                p = cnt / sz
                if p > 0:
                    c_entropy -= p * np.log2(p)
            total_entropy += (sz / clustered_movies) * c_entropy
        purity = pure_count / clustered_movies
        entropy = total_entropy

    clustered_ratio = clustered_movies / num_movies if num_movies else 0.0
    largest_ratio = largest_cluster / num_movies if num_movies else 0.0
    selection_score = purity * clustered_ratio * (1.0 - largest_ratio)
    return {
        "purity": purity,
        "entropy": entropy,
        "clustered_ratio": clustered_ratio,
        "largest_component_ratio": largest_ratio,
        "num_clusters": float(len(clusters)),
        "num_non_singleton_clusters": float(len(non_singleton)),
        "largest_cluster_size": float(largest_cluster),
        "selection_score": selection_score,
    }


def main() -> None:
    
    ensure_project_dirs()
    set_reproducible_seed()

    print("=" * 72)
    print("TASK 6: RECOMMENDATION BY WALKING RATINGS GRAPH")
    print("=" * 72)

    ratings, movies = load_movielens()
    user_to_idx, idx_to_user, movie_to_idx, idx_to_movie = create_mappings(ratings, movies)
    num_users = len(user_to_idx)
    num_movies = len(movie_to_idx)

    train_df, test_df = temporal_split(ratings, test_ratio=0.2)
    train_user_watched = build_user_movie_sets(train_df, user_to_idx, movie_to_idx)
    test_user_movies = build_user_movie_sets(test_df, user_to_idx, movie_to_idx)

    graph = build_bipartite_graph(train_df)
    num_user_nodes = sum(1 for node in graph if str(node).startswith("u"))
    num_movie_nodes = sum(1 for node in graph if str(node).startswith("m"))
    print(f"graph   {num_user_nodes} users + {num_movie_nodes} movies, {graph.number_of_edges()} edges")

    train_matrix = build_user_movie_matrix(train_df, user_to_idx, movie_to_idx, shape=(num_users, num_movies))
    train_binary = build_user_movie_matrix(
        train_df,
        user_to_idx,
        movie_to_idx,
        shape=(num_users, num_movies),
        binary=True,
    )

    popularity_counts = train_df["movieId"].value_counts()
    popular_movies = [int(mid) for mid in popularity_counts.index.tolist()]

    sample_user_idx = 0
    bfs_sample = bfs_recommend(graph, sample_user_idx, idx_to_user, movie_to_idx, train_user_watched, top_n=3, popular_fallback=popular_movies)
    ppr_scores = ppr_scores_from_matrix(train_binary, alpha=0.85, odd_steps=5)
    ppr_sample = ppr_recommend_from_scores(ppr_scores, sample_user_idx, train_user_watched, top_n=3, popular_fallback=popular_movies, movie_to_idx=movie_to_idx)

    print(f"\nbfs_recommend(user=0, n=3)   {[idx_to_movie[idx] for idx in bfs_sample]}")
    for idx, movie_idx in enumerate(bfs_sample, start=1):
        print(f"  {idx}. {get_movie_title(movies, idx_to_movie, movie_idx)}")

    print(f"\nppr_recommend(user=0, n=3)   {[idx_to_movie[idx] for idx in ppr_sample]}")
    for idx, movie_idx in enumerate(ppr_sample, start=1):
        print(f"  {idx}. {get_movie_title(movies, idx_to_movie, movie_idx)}")

    def bfs_fn(user_idx: int) -> list[int]:
        return bfs_recommend(graph, user_idx, idx_to_user, movie_to_idx, train_user_watched, top_n=TOP_K, popular_fallback=popular_movies)

    def ppr_fn(user_idx: int) -> list[int]:
        return ppr_recommend_from_scores(ppr_scores, user_idx, train_user_watched, top_n=TOP_K, popular_fallback=popular_movies, movie_to_idx=movie_to_idx)

    bfs_metrics = evaluate_recommender(bfs_fn, test_user_movies, num_users, num_movies, k=TOP_K)
    ppr_metrics = evaluate_recommender(ppr_fn, test_user_movies, num_users, num_movies, k=TOP_K)

    popularity_metrics = popularity_recommender_metrics(
        train_user_watched,
        test_user_movies,
        popular_movies,
        movie_to_idx,
        num_users,
        num_movies,
    )

    print("\nGraph recommender comparison:")
    print(f"{'Method':<24} {'Precision@10':>12} {'Recall@10':>10} {'NDCG@10':>9} {'Coverage':>9}")
    print("-" * 70)
    recommender_rows = [
        {"method": "Popularity", **popularity_metrics},
        {"method": "BFS Graph Walk", **bfs_metrics},
        {"method": "Personalized PageRank", **ppr_metrics},
    ]
    for row in recommender_rows:
        print(
            f"{row['method']:<24} {row['precision@10']:12.4f} "
            f"{row['recall@10']:10.4f} {row['ndcg@10']:9.4f} {row['coverage']:9.4f}"
        )

    print("\nSearching movie-cluster similarity thresholds...")
    S = build_item_similarity(train_matrix)
    genre_labels = primary_genre_by_movie(movies, movie_to_idx)
    cluster_rows: list[dict[str, float]] = []
    clusters_by_threshold: dict[float, list[list[int]]] = {}
    for threshold in GRAPH_SIMILARITY_THRESHOLDS:
        clusters = clusters_from_similarity(S, threshold, num_movies)
        metrics = cluster_quality(clusters, genre_labels, num_movies)
        row = {"threshold": threshold, **metrics}
        cluster_rows.append(row)
        clusters_by_threshold[threshold] = clusters
        print(
            f"threshold={threshold:.1f} purity={metrics['purity']:.4f} "
            f"entropy={metrics['entropy']:.4f} "
            f"largest_ratio={metrics['largest_component_ratio']:.4f} "
            f"clustered_ratio={metrics['clustered_ratio']:.4f} "
            f"score={metrics['selection_score']:.4f}"
        )

    best_cluster_row = max(cluster_rows, key=lambda row: row["selection_score"])
    best_threshold = float(best_cluster_row["threshold"])
    print(f"\nSelected cluster threshold={best_threshold:.1f}")

    selected_clusters = sorted(
        [cluster for cluster in clusters_by_threshold[best_threshold] if len(cluster) > 1],
        key=len,
        reverse=True,
    )
    cluster_examples: list[dict[str, object]] = []
    for cluster in selected_clusters[:5]:
        titles = [get_movie_title(movies, idx_to_movie, movie_idx) for movie_idx in cluster[:5]]
        labels = Counter(genre_labels.get(movie_idx, "unknown") for movie_idx in cluster)
        c_entropy = 0.0
        sz = len(cluster)
        for cnt in labels.values():
            p = cnt / sz
            if p > 0:
                c_entropy -= p * np.log2(p)
        cluster_examples.append(
            {
                "size": len(cluster),
                "dominant_genre": labels.most_common(1)[0][0],
                "purity": labels.most_common(1)[0][1] / len(cluster),
                "entropy": c_entropy,
                "sample_titles": titles,
            }
        )
    print("\nSelected cluster examples:")
    for idx, example in enumerate(cluster_examples, start=1):
        print(
            f"Group {idx} size={example['size']} dominant={example['dominant_genre']} "
            f"purity={example['purity']:.3f} entropy={example['entropy']:.3f}"
        )
        print(f"  Sample titles: {example['sample_titles']}")

    write_json(
        OUTPUT_DIR / "graph_results.json",
        {
            "graph": {
                "user_nodes": num_user_nodes,
                "movie_nodes": num_movie_nodes,
                "edges": graph.number_of_edges(),
            },
            "recommender_metrics": recommender_rows,
            "cluster_thresholds": cluster_rows,
            "selected_cluster_threshold": best_threshold,
            "cluster_examples": cluster_examples,
        },
    )
    write_csv(OUTPUT_DIR / "graph_cluster_thresholds.csv", cluster_rows)


if __name__ == "__main__":
    main()
