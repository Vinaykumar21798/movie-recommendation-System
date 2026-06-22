# Movie Recommendation System

Week 4 recommender-systems project using MovieLens `ml-latest-small`.

## Setup

```powershell
pip install -r requirements.txt
```

## Run Core Tasks

```powershell
python load_data.py
python recommender_baseline.py
python recommender_item_cf.py
python recommender_lsh.py --skip_brute_force
python recommender_mf.py
python recommender_graph.py
python recommender_diversity_coldstart.py
```

## Run API

```powershell
python recommender_api.py --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

Example:

```text
http://127.0.0.1:8000/recommend/1?n=10&method=mf
```

Methods:

- `popularity`
- `item_cf`
- `mf`
- `graph`

Health check:

```text
http://127.0.0.1:8000/health
```

## Validation

```powershell
python -m pytest tests -q
python benchmark.py
```

Use the full benchmark only when you have time for the heavier LSH run:

```powershell
python benchmark.py --full
```

## Artifacts

- `artifacts/`: persisted model and plots
- `outputs/`: metrics, benchmark CSV/JSON files, final report

## Leakage Policy

- Item-CF thresholds are selected on validation only.
- MF factors are selected on validation only.
- MMR lambda is selected on validation only.
- Final test metrics are computed once after selection.
- API serving filters all known rated movies for a user.

