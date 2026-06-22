# Production-Grade Movie Recommendation System

An end-to-end, high-performance Movie Recommendation System built on MovieLens `ml-latest-small`. The project implements multiple recommendation strategies, hybrid fallbacks, similarity indices, locality-sensitive hashing, graph-based walks, diversity reranking, and an API serving layer.

---

## Key Features & Algorithms

The system implements 8 progressive tasks designed to move from simple popularity-based recommenders to complex matrix factorization, graph networks, and high-performance caching services:

1. **Reversible Mapping & CSR Representation** (Task 1): Encodes sparse user/movie identifiers into contiguous indices. Formulates ratings as memory-efficient SciPy CSR matrices ($U \times I$) and fast-looping adjacency lists.
2. **Temporal Split & Evaluation Baseline** (Task 2): Implements chronological temporal train/validation/test partitions per user to prevent future data leakage. Computes Precision@K, Recall@K, NDCG@K, and Catalog Coverage.
3. **Sparse Item Collaborative Filtering** (Task 3): Computes sparse item cosine similarity in one matrix pass to minimize memory footprint. Grabs nearest neighbors efficiently via heaps (`heapq.nlargest`).
4. **Locality-Sensitive Hashing (LSH)** (Task 4): Generates 128-permutation MinHash signatures and indexes them into LSH buckets to achieve a **$34.0\times$ candidate pair comparison reduction** vs brute force.
5. **Latent Matrix Factorization SVD** (Task 5): Decomposes ratings into latent user and item factor spaces. Plots factor sweep tuning on validation to select the optimal model size ($K=50$).
6. **Bipartite Graph Network Walks** (Task 6): Links users and movies in a bipartite network. Recommends items via 3-hop BFS paths and Personalized PageRank (PPR) random walks. Clusters movies via Union-Find and computes Cluster Purity and Cluster Entropy.
7. **MMR Diversity Reranking & Cold Start** (Task 7): Applies Maximal Marginal Relevance (MMR) to balance relevance vs genre similarity. Implements content-based user and movie cold start handlers using genre overlap.
8. **Asynchronous FastAPI Serving & LRU Caching** (Task 8): Exposes endpoints with strict Pydantic schemas. Pre-loads models on startup and leverages `@functools.lru_cache` to achieve a **$9,000\times$ request speedup** ($13.5$ms fresh request vs $0.0015$ms cached).

---

## Project Structure

```text
├── artifacts/              # Persisted API model binaries (pickle) and validation plots
├── outputs/                # Evaluation reports, JSON benchmark files, and CSV sweeps
│   └── reports/            # Consolidated final report
├── tests/                  # Pytest unit and integration files
├── config.py               # Shared global constants, logger setup, and directory creators
├── recommender_common.py   # Common data loader, temporal splits, and metric calculators
├── load_data.py            # Task 1: Mapping & matrix statistics verification
├── recommender_baseline.py # Task 2: Popularity baseline evaluation & k-sweep
├── recommender_item_cf.py  # Task 3: Item Cosine Collaborative Filtering & padding fallbacks
├── recommender_lsh.py      # Task 4: MinHash LSH similarity comparison benchmark
├── recommender_mf.py       # Task 5: SVD Taste-vector matrix factorization
├── recommender_graph.py    # Task 6: Bipartite Graph BFS, PPR walks, and Union-Find clustering
├── recommender_diversity_coldstart.py # Task 7: MMR diversity reranking & Cold Start handlers
├── recommender_api.py      # Task 8: FastAPI serving endpoints & cache benchmark
├── benchmark.py            # Project benchmark and report generator
├── requirements.txt        # Production dependency pinning
└── README.md               # Documentation
```

---

## Getting Started

### 1. Installation & Environment Setup

Clone the repository and install the production dependencies:

```powershell
pip install -r requirements.txt
```

### 2. Run Individual Recommender Tasks

You can run each algorithm module independently to inspect console validation prints and output files:

```powershell
# Task 1: Load MovieLens, construct maps, and print facts
python load_data.py

# Task 2: Evaluate popular baseline metrics and sweep K
python recommender_baseline.py

# Task 3: Build sparse cosine similarity and evaluate Item-CF
python recommender_item_cf.py

# Task 4: Run Locality-Sensitive Hashing benchmark vs brute force
python recommender_lsh.py --skip_brute_force

# Task 5: Train Matrix Factorization SVD and output nearest movies
python recommender_mf.py

# Task 6: Execute Graph walks (BFS/PPR) and similarity clustering
python recommender_graph.py

# Task 7: Run MMR diversity reranking sweeps and Cold Start validators
python recommender_diversity_coldstart.py
```

### 3. Run FastAPI Serving Layer

To boot the FastAPI server locally:

```powershell
python recommender_api.py --port 8000
```

*   **Interactive Documentation**: Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser.
*   **User Recommendations**: Query `GET /recommend/{user_id}`.
    *   *Example*: [http://127.0.0.1:8000/recommend/1?n=10&method=mf](http://127.0.0.1:8000/recommend/1?n=10&method=mf)
*   **Method Comparison**: Get evaluation metrics for all recommenders: [http://127.0.0.1:8000/compare](http://127.0.0.1:8000/compare)
*   **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## Validation & Benchmarking

### Run Automated Tests
We maintain integration and unit test coverage over recommendation correctness, temporal splits, cluster quality, and API routing schemas:

```powershell
python -m pytest
```

### Run Full Benchmark Runner
To automate the training, execution, and comparison of all Tasks 1-8 in one pass, run the project benchmark runner:

```powershell
python benchmark.py --full
```

This compiles a Markdown audit report saved at [outputs/reports/final_evaluation_report.md](file:///c:/Users/lenovo/Desktop/Data%20Factz%20Projects/Movie%20Recommendation%20System/outputs/reports/final_evaluation_report.md).

---

## Strict Evaluation & Data Leakage Policy

1.  **Temporal Splitting**: Model training is strictly chronologically partitioned. No ratings from future timestamps leak into similarities, factors, or network walks.
2.  **Validation-Only Hyperparameter Tuning**: All parameters (collaborative filtering cutoffs, latent factor dimension sizing, similarity cluster boundaries, and MMR diversification lambdas) are swept and selected using the validation set. Final evaluations are computed once on the held-out test split.
3.  **Corrected Average Metrics**: Evaluation metrics (precision, recall, NDCG) are calculated only over users who have active ground truth test items, preventing dilution from train-only users.
4.  **Production Rating Masking**: The `/recommend` serving layer filters out all historic ratings (both train and test sets) from the output payloads to ensure users are never recommended movies they have already watched.
