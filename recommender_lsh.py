"""Task 4: Cross-platform MinHash LSH movie similarity benchmark."""

from __future__ import annotations

import argparse
import time
import tracemalloc
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from datasketch import MinHash, MinHashLSH

from config import LSH_THRESHOLDS, OUTPUT_DIR, ensure_project_dirs, set_reproducible_seed
from recommender_common import download_and_extract_dataset, write_csv, write_json


def load_movie_user_data(dataset_path: str) -> tuple[dict[int, set[int]], list[int], pd.DataFrame, pd.DataFrame]:
    """Load ratings and represent each rated movie as a set of user indices."""
    ratings = pd.read_csv(f"{dataset_path}/ratings.csv")
    movies = pd.read_csv(f"{dataset_path}/movies.csv")

    unique_movies = sorted(ratings["movieId"].unique())
    movie_to_idx = {int(movie_id): idx for idx, movie_id in enumerate(unique_movies)}
    unique_users = sorted(ratings["userId"].unique())
    user_to_idx = {int(user_id): idx for idx, user_id in enumerate(unique_users)}

    movie_users: dict[int, set[int]] = {}
    grouped = ratings.groupby("movieId")["userId"].apply(set).reset_index()
    for row in grouped.itertuples(index=False):
        movie_users[movie_to_idx[int(row.movieId)]] = {user_to_idx[int(uid)] for uid in row.userId}

    return movie_users, [int(m) for m in unique_movies], movies, ratings


def compute_exact_jaccard_sparse(
    movie_users: dict[int, set[int]],
    num_movies: int,
    num_users: int,
    k: int,
) -> tuple[dict[int, list[tuple[float, int]]], float]:
    """Compute exact top-k Jaccard neighbors using sparse matrix multiplication."""
    row_indices: list[int] = []
    col_indices: list[int] = []
    for movie_idx, users in movie_users.items():
        for user_idx in users:
            row_indices.append(movie_idx)
            col_indices.append(user_idx)

    A = sp.csr_matrix(
        (np.ones(len(row_indices), dtype=np.uint8), (row_indices, col_indices)),
        shape=(num_movies, num_users),
    )
    intersections = A.dot(A.T).tocoo()
    movie_degrees = np.asarray(A.sum(axis=1)).ravel()

    mask = intersections.row < intersections.col
    rows = intersections.row[mask]
    cols = intersections.col[mask]
    intersection_values = intersections.data[mask].astype(np.float32)
    union_values = movie_degrees[rows] + movie_degrees[cols] - intersection_values
    jaccard_values = intersection_values / union_values

    J = sp.csr_matrix((jaccard_values, (rows, cols)), shape=(num_movies, num_movies))
    J = (J + J.T).tocsr()
    exact_memory_mb = (J.data.nbytes + J.indices.nbytes + J.indptr.nbytes) / (1024 * 1024)

    exact_neighbors: dict[int, list[tuple[float, int]]] = {}
    for movie_idx in range(num_movies):
        start, end = J.indptr[movie_idx], J.indptr[movie_idx + 1]
        data = J.data[start:end]
        indices = J.indices[start:end]
        if len(data) > k:
            top_idx = np.argpartition(data, -k)[-k:]
            top_data = data[top_idx]
            top_cols = indices[top_idx]
            order = np.argsort(top_data)[::-1]
            exact_neighbors[movie_idx] = [(float(top_data[i]), int(top_cols[i])) for i in order]
        else:
            order = np.argsort(data)[::-1]
            exact_neighbors[movie_idx] = [(float(data[i]), int(indices[i])) for i in order]

    return exact_neighbors, exact_memory_mb


def run_brute_force_jaccard_benchmark(movie_users: dict[int, set[int]], num_movies: int) -> float:
    """Benchmark a partial naive all-pairs loop and extrapolate full runtime."""
    sample_size = min(1000, num_movies)
    t0 = time.perf_counter()
    for i in range(sample_size):
        users_i = movie_users[i]
        len_i = len(users_i)
        for j in range(i + 1, num_movies):
            users_j = movie_users[j]
            intersection = len(users_i & users_j)
            if intersection:
                _ = intersection / (len_i + len(users_j) - intersection)
    elapsed = time.perf_counter() - t0
    return elapsed * (num_movies / sample_size)


def build_minhashes(movie_users: dict[int, set[int]], num_perm: int) -> dict[int, MinHash]:
    """Build MinHash signatures for every movie."""
    minhashes: dict[int, MinHash] = {}
    for movie_idx, users in movie_users.items():
        minhash = MinHash(num_perm=num_perm)
        for user_idx in users:
            minhash.update(str(user_idx).encode("utf-8"))
        minhashes[movie_idx] = minhash
    return minhashes


def evaluate_lsh_threshold(
    threshold: float,
    minhashes: dict[int, MinHash],
    movie_users: dict[int, set[int]],
    exact_neighbors: dict[int, list[tuple[float, int]]],
    num_movies: int,
    total_possible_pairs: int,
    num_perm: int,
    k: int,
) -> dict[str, float]:
    """Build/query one LSH index and compute recall, runtime, comparisons, and memory."""
    tracemalloc.start()
    t0 = time.perf_counter()

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for movie_idx, minhash in minhashes.items():
        lsh.insert(movie_idx, minhash)
    build_seconds = time.perf_counter() - t0

    query_start = time.perf_counter()
    candidate_pairs: set[tuple[int, int]] = set()
    retrieved_by_movie: dict[int, set[int]] = {}
    for movie_idx in range(num_movies):
        candidates = set(int(c) for c in lsh.query(minhashes[movie_idx]))
        retrieved_by_movie[movie_idx] = candidates
        for candidate_idx in candidates:
            if movie_idx < candidate_idx:
                candidate_pairs.add((movie_idx, candidate_idx))
    query_seconds = time.perf_counter() - query_start

    eval_start = time.perf_counter()
    lsh_similarities: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for movie_a, movie_b in candidate_pairs:
        users_a = movie_users[movie_a]
        users_b = movie_users[movie_b]
        intersection = len(users_a & users_b)
        jaccard = intersection / (len(users_a) + len(users_b) - intersection)
        lsh_similarities[movie_a].append((jaccard, movie_b))
        lsh_similarities[movie_b].append((jaccard, movie_a))
    eval_seconds = time.perf_counter() - eval_start

    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    recall_scores: list[float] = []
    for movie_idx in range(num_movies):
        true_set = {neighbor_idx for sim, neighbor_idx in exact_neighbors.get(movie_idx, [])[:k] if sim > 0}
        if not true_set:
            continue
        retrieved_set = retrieved_by_movie.get(movie_idx, set())
        recall_scores.append(len(true_set & retrieved_set) / len(true_set))

    runtime_seconds = build_seconds + query_seconds + eval_seconds
    candidate_count = len(candidate_pairs)
    comparisons_pct = (candidate_count / total_possible_pairs) * 100 if total_possible_pairs else 0.0
    recall = float(np.mean(recall_scores)) if recall_scores else 0.0

    return {
        "threshold": threshold,
        "candidate_pairs": float(candidate_count),
        "all_pairs": float(total_possible_pairs),
        "comparisons_pct": comparisons_pct,
        "recall": recall,
        "runtime_seconds": runtime_seconds,
        "build_seconds": build_seconds,
        "query_seconds": query_seconds,
        "similarity_eval_seconds": eval_seconds,
        "peak_memory_mb": peak_bytes / (1024 * 1024),
        "threshold_runtime_score": recall / runtime_seconds if runtime_seconds else 0.0,
        "pair_reduction_x": total_possible_pairs / max(1, candidate_count),
    }


def main() -> None:
    """Run the LSH benchmark and automatically select the best threshold."""
    parser = argparse.ArgumentParser(description="LSH similarity benchmark for movies")
    parser.add_argument("--num_perm", type=int, default=128)
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(LSH_THRESHOLDS))
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--skip_brute_force", action="store_true")
    args = parser.parse_args()

    ensure_project_dirs()
    set_reproducible_seed()

    print("=" * 72)
    print("TASK 4: LSH MOVIE SIMILARITY SEARCH")
    print("=" * 72)
    print(f"Parameters: num_perm={args.num_perm}, thresholds={args.thresholds}, k={args.k}")

    dataset_path = download_and_extract_dataset()
    movie_users, unique_movies, _, ratings = load_movie_user_data(dataset_path)
    num_movies = len(unique_movies)
    num_users = int(ratings["userId"].nunique())
    total_possible_pairs = num_movies * (num_movies - 1) // 2
    print(f"Loaded {num_movies} rated movies and {num_users} users.")

    print("\nComputing optimized exact Jaccard top-k baseline...")
    t_exact = time.perf_counter()
    exact_neighbors, exact_memory_mb = compute_exact_jaccard_sparse(movie_users, num_movies, num_users, args.k)
    exact_runtime = time.perf_counter() - t_exact
    print(f"Optimized exact runtime: {exact_runtime:.3f}s, memory: {exact_memory_mb:.2f} MB")

    brute_force_runtime = None
    if not args.skip_brute_force:
        brute_force_runtime = run_brute_force_jaccard_benchmark(movie_users, num_movies)
        print(f"Extrapolated naive brute-force runtime: {brute_force_runtime:.1f}s")

    print("\nBuilding MinHash signatures once for all thresholds...")
    t_sig = time.perf_counter()
    minhashes = build_minhashes(movie_users, args.num_perm)
    signature_runtime = time.perf_counter() - t_sig
    print(f"Signature runtime: {signature_runtime:.3f}s")

    benchmark_rows: list[dict[str, float]] = []
    for threshold in args.thresholds:
        print(f"\nBenchmarking threshold={threshold:.1f}...")
        row = evaluate_lsh_threshold(
            threshold,
            minhashes,
            movie_users,
            exact_neighbors,
            num_movies,
            total_possible_pairs,
            args.num_perm,
            args.k,
        )
        row["signature_seconds"] = signature_runtime
        row["total_lsh_seconds"] = row["runtime_seconds"] + signature_runtime
        row["score_recall_per_second"] = row["recall"] / max(row["total_lsh_seconds"], 1e-9)
        row["runtime_speedup_vs_optimized_exact"] = exact_runtime / max(row["total_lsh_seconds"], 1e-9)
        if brute_force_runtime is not None:
            row["runtime_speedup_vs_naive_bruteforce"] = brute_force_runtime / max(row["total_lsh_seconds"], 1e-9)
        benchmark_rows.append(row)

    best_row = max(benchmark_rows, key=lambda row: row["score_recall_per_second"])

    print("\nBenchmark table:")
    print(
        f"{'thr':>5} {'candidates':>12} {'cmp%':>9} {'recall':>8} "
        f"{'lsh_s':>9} {'mem_mb':>9} {'score':>10}"
    )
    for row in benchmark_rows:
        print(
            f"{row['threshold']:5.1f} {int(row['candidate_pairs']):12,} "
            f"{row['comparisons_pct']:9.4f} {row['recall']:8.4f} "
            f"{row['total_lsh_seconds']:9.2f} {row['peak_memory_mb']:9.2f} "
            f"{row['score_recall_per_second']:10.6f}"
        )

    print(
        f"\nSelected threshold={best_row['threshold']:.1f} "
        f"using score=recall/runtime ({best_row['score_recall_per_second']:.6f})."
    )
    print(
        f"Optimized exact remained {'faster' if best_row['runtime_speedup_vs_optimized_exact'] < 1 else 'slower'} "
        f"on this small dataset; LSH still reduces pair comparisons by "
        f"{best_row['pair_reduction_x']:.1f}x at the selected threshold."
    )

    csv_rows = [{k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()} for row in benchmark_rows]
    write_csv(OUTPUT_DIR / "lsh_benchmark.csv", csv_rows)
    write_json(
        OUTPUT_DIR / "lsh_benchmark.json",
        {
            "num_perm": args.num_perm,
            "k": args.k,
            "optimized_exact_runtime_seconds": exact_runtime,
            "optimized_exact_memory_mb": exact_memory_mb,
            "brute_force_runtime_seconds": brute_force_runtime,
            "rows": benchmark_rows,
            "selected_threshold": best_row["threshold"],
            "selection_rule": "max(recall / total_lsh_seconds)",
        },
    )


if __name__ == "__main__":
    main()
