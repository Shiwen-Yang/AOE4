"""LightGBM training for anonymous opponent civ-choice prediction."""
from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from .anonymous_features import (
    ANONYMOUS_CONTEXT_FEATURES,
    ANONYMOUS_FEATURES,
    prepare_anonymous_X,
)
from .model import DEFAULT_PARAMS


def train_anonymous_lgbm(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    params: dict | None = None,
) -> lgb.LGBMClassifier:
    p = {**DEFAULT_PARAMS, **(params or {})}
    n_est = p.pop("n_estimators", 500)

    X_train = prepare_anonymous_X(train_df)
    y_train = train_df["target"].values
    X_valid = prepare_anonymous_X(valid_df)
    y_valid = valid_df["target"].values

    print("\n=== Training Anonymous LightGBM ===")
    print(f"  Features: {len(ANONYMOUS_FEATURES)}")
    print(f"  Train rows: {len(X_train):,}  |  Valid rows: {len(X_valid):,}")
    print(f"  Positive rate (train): {y_train.mean():.4f}")

    model = lgb.LGBMClassifier(n_estimators=n_est, **p)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        categorical_feature=ANONYMOUS_CONTEXT_FEATURES,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    return model
