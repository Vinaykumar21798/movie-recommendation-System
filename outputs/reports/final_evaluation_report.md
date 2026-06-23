# Week 4 Movie Recommendation System Remediation Report

Mode: full task benchmark
Final status: PASS

| Check | Exit Code | Seconds |
|---|---:|---:|
| recommender_baseline.py | 0 | 4.53 |
| recommender_item_cf.py | 0 | 27.52 |
| recommender_lsh.py --skip_brute_force | 0 | 193.86 |
| recommender_mf.py | 0 | 79.87 |
| recommender_graph.py | 0 | 38.01 |
| recommender_diversity_coldstart.py | 0 | 82.30 |
| recommender_api.py --eval-only | 0 | 32.31 |

Validation scope:
- Leakage-safe train/validation/test tuning for Item-CF, MF, and MMR.
- Cross-platform LSH output and dependency coverage.
- Portable outputs/ and artifacts/ directories.
- BFS and Personalized PageRank graph metrics.
- FastAPI schema validation, cache benchmark, and endpoint tests.