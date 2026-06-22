

from __future__ import annotations

from collections import defaultdict

from recommender_common import (
    build_user_movie_matrix,
    create_mappings,
    download_and_extract_dataset,
    load_movielens,
)


def main() -> None:
    
    ratings, movies = load_movielens()
    user_to_idx, idx_to_user, movie_to_idx, idx_to_movie = create_mappings(ratings, movies)

    row_indices = ratings["userId"].map(user_to_idx).to_numpy()
    col_indices = ratings["movieId"].map(movie_to_idx).to_numpy()
    ratings_values = ratings["rating"].to_numpy()

    num_users = len(user_to_idx)
    num_movies = len(movie_to_idx)
    matrix = build_user_movie_matrix(ratings, user_to_idx, movie_to_idx, shape=(num_users, num_movies))

    adjacency: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for user_idx, movie_idx, rating in zip(row_indices, col_indices, ratings_values):
        adjacency[int(user_idx)].append((int(movie_idx), float(rating)))

    print("\n--- DATA STATISTICS ---")
    user_to_idx_snippet = {key: user_to_idx[key] for key in list(user_to_idx.keys())[:5]}
    print(f"user_to_idx   {user_to_idx_snippet} (truncated)...")
    sample_adjacency = sorted(adjacency.get(0, []))[:3]
    print(f"adjacency[0]  {sample_adjacency}   # user 0's movies")

    sparsity = (1.0 - matrix.nnz / (num_users * num_movies)) * 100
    print(f"matrix        shape {matrix.shape}   {sparsity:.1f}% empty")

    busiest_user_idx = max(adjacency.keys(), key=lambda key: len(adjacency[key]))
    busiest_user_original_id = idx_to_user[busiest_user_idx]
    busiest_user_ratings_count = len(adjacency[busiest_user_idx])

    top_movie_counts = ratings["movieId"].value_counts()
    top_movie_id = int(top_movie_counts.index[0])
    top_movie_ratings_count = int(top_movie_counts.iloc[0])
    top_movie_title = movies[movies["movieId"] == top_movie_id]["title"].iloc[0]

    print(
        f"busiest user  {busiest_user_original_id} ({busiest_user_ratings_count} ratings)   "
        f"top movie  {top_movie_title} ({top_movie_ratings_count})"
    )
    print("-----------------------\n")

    print("Running verification checks...")
    test_user_id = next(iter(user_to_idx))
    assert idx_to_user[user_to_idx[test_user_id]] == test_user_id, "User ID mapping failed"
    test_movie_id = next(iter(movie_to_idx))
    assert idx_to_movie[movie_to_idx[test_movie_id]] == test_movie_id, "Movie ID mapping failed"

    user_movies_dict = sorted(adjacency[busiest_user_idx])
    user_row = matrix[busiest_user_idx]
    _, sparse_cols = user_row.nonzero()
    user_movies_matrix = sorted(zip([int(col) for col in sparse_cols], [float(v) for v in user_row.data]))

    assert len(user_movies_dict) == len(user_movies_matrix), "CSR row and adjacency rating counts differ"
    for (dict_movie, dict_rating), (matrix_movie, matrix_rating) in zip(user_movies_dict, user_movies_matrix):
        assert dict_movie == matrix_movie, "CSR row and adjacency movie indices differ"
        assert abs(dict_rating - matrix_rating) < 1e-9, "CSR row and adjacency ratings differ"

    print(f"Verification check for user idx {busiest_user_idx} passed successfully.")


if __name__ == "__main__":
    main()
