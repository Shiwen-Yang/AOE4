"""
Dataset orchestration. Thin wrapper around features.build_dataset so callers have a
stable `dataset.build_dataset(...)` entry point (mirrors the sibling packages' layout).
"""
from .features import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_dataset

__all__ = ["build_dataset", "ALL_FEATURES", "CATEGORICAL_FEATURES", "NUMERIC_FEATURES"]
