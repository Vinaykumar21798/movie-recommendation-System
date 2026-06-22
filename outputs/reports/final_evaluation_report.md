# Week 4 Movie Recommendation System Remediation Report

Mode: full task benchmark
Final status: PASS

| Check | Exit Code | Seconds |
|---|---:|---:|
| recommender_item_cf.py | 0 | 23.77 |
| recommender_lsh.py --skip_brute_force | 0 | 134.90 |
| recommender_mf.py | 0 | 32.75 |
| recommender_graph.py | 0 | 38.24 |
| recommender_diversity_coldstart.py | 0 | 56.32 |
| recommender_api.py --eval-only | 0 | 30.30 |

Validation scope:
- Leakage-safe train/validation/test tuning for Item-CF, MF, and MMR.
- Cross-platform LSH output and dependency coverage.
- Portable outputs/ and artifacts/ directories.
- BFS and Personalized PageRank graph metrics.
- FastAPI schema validation, cache benchmark, and endpoint tests.