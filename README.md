# Production-Grade Movie Recommendation System

An end-to-end, high-performance Movie Recommendation System built on MovieLens `ml-latest-small`. The project implements multiple recommendation strategies, hybrid fallbacks, similarity indices, locality-sensitive hashing, graph-based walks, diversity reranking, a central Model Registry, and an API serving layer.

---

## Key Features & Algorithms

The system implements 8 progressive tasks designed to move from simple popularity-based recommenders to complex matrix factorization, graph networks, and high-performance caching services:

1. **Reversible Mapping & CSR Representation** (Task 1): Encodes sparse user/movie identifiers into contiguous indices. Formulates ratings as memory-efficient SciPy CSR matrices ($U \times I$) and fast-looping adjacency lists.
2. **Standardized Split & Caching** (Task 2): Implements chronological temporal train/validation/test partitions per user (60/20/20 ratio) to prevent future data leakage. Splits are cached on disk as CSVs to guarantee identical validation cohorts across all modules.
3. **Sparse Item Collaborative Filtering** (Task 3): Computes sparse item cosine similarity in one matrix pass to minimize memory footprint. Grabs nearest neighbors efficiently via heaps (`heapq.nlargest`).
4. **Locality-Sensitive Hashing (LSH)** (Task 4): Generates 128-permutation MinHash signatures and indexes them into LSH buckets to achieve a **$34.0\times$ candidate pair comparison reduction** vs brute force.
5. **Latent Matrix Factorization SVD** (Task 5): Decomposes ratings into latent user and item factor spaces. Plots factor sweep tuning on validation to select the optimal model size ($K=50$).
6. **Bipartite Graph Network Walks** (Task 6): Links users and movies in a bipartite network. Recommends items via 3-hop BFS paths and Personalized PageRank (PPR) random walks. Clusters movies via Union-Find and computes Cluster Purity and Cluster Entropy.
7. **MMR Diversity Reranking & Cold Start** (Task 7): Applies Maximal Marginal Relevance (MMR) to balance relevance vs genre similarity. Implements content-based user and movie cold start handlers using genre overlap.
8. **Model Registry & Serving Layer** (Task 8): Implements a central registry managing serialization, hyperparameter states, and offline test metrics. The serving layer loads pre-trained states from the registry and exposes endpoints with strict Pydantic schemas and `@functools.lru_cache` (caching provides a **$9,000\times$ request speedup**).

---

## Project Structure

```text
├── artifacts/              # Persisted model binaries (pickle) and validation plots managed by ModelRegistry
├── outputs/                # Evaluation reports, JSON benchmark files, and CSV sweeps
│   └── reports/            # Consolidated final report
├── data/                   # Raw MovieLens data and cached train/val/test CSV splits
├── tests/                  # Pytest unit and integration files
├── config.py               # Shared global constants, logger setup, and directory creators
├── recommender_common.py   # Data loader, cached split loader, metrics engine, and ModelRegistry
├── load_data.py            # Task 1: Mapping & matrix statistics verification
├── recommender_baseline.py # Task 2: Popularity baseline evaluation & registers popularity model
├── recommender_item_cf.py  # Task 3: Item Cosine Collaborative Filtering & registers item_cf model
├── recommender_lsh.py      # Task 4: MinHash LSH similarity comparison benchmark
├── recommender_mf.py       # Task 5: Matrix Factorization SVD & registers SVD model
├── recommender_graph.py    # Task 6: Graph BFS, PPR walks, and registers graph model
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

### 2. Run Recommender Training & Register Models

To populate the **Model Registry**, you must run the model training modules first. These scripts automatically load the cached train/val/test split, train the parameters, evaluate test performance, and save model payloads:

```powershell
# Task 1: Load MovieLens, construct maps, and print facts
python load_data.py

# Task 2: Evaluate popular baseline metrics and register popularity model
python recommender_baseline.py

# Task 3: Build sparse cosine similarity and register Item-CF
python recommender_item_cf.py

# Task 4: Run Locality-Sensitive Hashing benchmark vs brute force
python recommender_lsh.py --skip_brute_force

# Task 5: Train SVD matrix factorization and register mf model
python recommender_mf.py

# Task 6: Execute Graph walks and register graph bipartite components
python recommender_graph.py

# Task 7: Run MMR diversity reranking sweeps and Cold Start validators
python recommender_diversity_coldstart.py
```

### 3. Run FastAPI Serving Layer

To boot the FastAPI server locally (the server will automatically load model configurations from the central registry):

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

To verify recommendation correctness, split temporal order, and API schema validations:

```powershell
python -m pytest
```

### Run Full Benchmark Runner

To automate the training, execution, registration, and comparison of all Tasks in one pass, run the project benchmark runner:

```powershell
python benchmark.py --full
```

This compiles a consolidated Markdown report saved at [outputs/reports/final_evaluation_report.md](file:///c:/Users/lenovo/Desktop/Data%20Factz%20Projects/Movie%20Recommendation%20System/outputs/reports/final_evaluation_report.md).

---

## Strict Evaluation & Data Leakage Policy

1.  **Temporal Splitting**: Model training is strictly chronologically partitioned. No ratings from future timestamps leak into similarities, factors, or network walks.
2.  **Shared Split Caching**: Ratings are split into static `train.csv`, `val.csv`, and `test.csv` cached files. This guarantees that model training, validation tuning, and production evaluation use identical evaluation users and ground truth splits.
3.  **Unified Model Registry**: Live models served by the API are strictly read from the registry, preventing on-the-fly retraining or data leakage. Calling `force_rebuild` on the API endpoint raises an exception to block live retraining.
4.  **Production Rating Masking**: The `/recommend` serving layer filters out all historic ratings (both train and test sets) from the output payloads to ensure users are never recommended movies they have already watched.
