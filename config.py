"""Shared configuration for the Movie Recommendation System project."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np


RANDOM_SEED = 42
TOP_K = 10

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
ARTIFACT_DIR = BASE_DIR / "artifacts"
REPORT_DIR = OUTPUT_DIR / "reports"

ITEM_CF_THRESHOLDS = (0.0, 3.0, 3.5, 4.0)
LSH_THRESHOLDS = (0.5, 0.3, 0.2)
MF_FACTORS = (10, 20, 50, 100, 150, 200)
GRAPH_SIMILARITY_THRESHOLDS = (0.5, 0.6, 0.7, 0.8)
MMR_LAMBDAS = (0.3, 0.5, 0.7)

TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2

API_ARTIFACT_PATH = ARTIFACT_DIR / "api_model_state.pkl"


def ensure_project_dirs() -> None:
    """Create portable output directories used by scripts and tests."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def set_reproducible_seed(seed: int = RANDOM_SEED) -> None:
    """Seed all project-level pseudo-random generators."""
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str) -> logging.Logger:
    """Return a consistently configured project logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)
