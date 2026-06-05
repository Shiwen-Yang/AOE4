"""
LightGBM and XGBoost model training and persistence.

Temporal split:
  - Train:      first (1 - valid_frac - test_frac) of the data by started_at
  - Validation: next valid_frac
  - Test:       last test_frac  (held out — only reported, never used for tuning)
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import MODEL_META_PATH, MODEL_PATH, TEST_FRAC, VALID_FRAC, XGB_META_PATH, XGB_MODEL_PATH
from .features_extra import ALL_EXTRA_FEATURES, P6_CATEGORICAL_FEATURES

# Features passed to LightGBM
# Categorical features are handled natively — LightGBM converts strings internally.
NUMERIC_FEATURES = [
    "mmr_a", "mmr_b", "mmr_diff",
    "rating_a", "rating_b", "rating_diff",
    "skill_a", "skill_b", "skill_diff",
    "missing_mmr_a", "missing_mmr_b",
    "missing_rating_a", "missing_rating_b",
    "missing_skill_a", "missing_skill_b",
    "games_lifetime_a", "wins_lifetime_a",
    "games_lifetime_b", "wins_lifetime_b",
    "games_season_a", "wins_season_a",
    "games_season_b", "wins_season_b",
    "games_diff", "wr_diff",
    "overall_wr_a", "overall_wr_b",
    "season_wr_a", "season_wr_b",
    "days_since_a", "days_since_b",
    "civ_games_a", "civ_wins_a", "civ_wr_a",
    "civ_games_b", "civ_wins_b", "civ_wr_b",
    "map_games_a", "map_wins_a", "map_wr_a",
    "map_games_b", "map_wins_b", "map_wr_b",
    "prior_matchup_games", "prior_matchup_wins", "prior_matchup_wr_a",
    "is_new_player_a", "is_new_player_b",
    "civs_known", "map_known", "full_context_known",
]

CATEGORICAL_FEATURES = [
    "civ_a", "civ_b", "map", "patch", "season",
] + P6_CATEGORICAL_FEATURES

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ALL_EXTRA_FEATURES

DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting": "gbdt",
    # Optuna-tuned hyperparameters (from models/lgbm_best_params.json)
    "num_leaves": 167,
    "min_child_samples": 126,
    "learning_rate": 0.028613301027563053,
    "feature_fraction": 0.6044477784794249,
    "bagging_fraction": 0.9243151079931942,
    "bagging_freq": 5,
    "lambda_l1": 3.5540285638395406,
    "lambda_l2": 3.1088199468129893,
    "n_estimators": 1000,
    "verbose": -1,
    "is_unbalance": False,
    "random_state": 42,
}


def _temporal_split(
    df: pd.DataFrame,
    valid_frac: float = VALID_FRAC,
    test_frac: float = TEST_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time: train | valid | test."""
    df_sorted = df.sort_values("started_at").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * (1 - valid_frac - test_frac))
    valid_end = int(n * (1 - test_frac))

    train = df_sorted.iloc[:train_end]
    valid = df_sorted.iloc[train_end:valid_end]
    test = df_sorted.iloc[valid_end:]
    return train, valid, test


def _make_dataset(
    df: pd.DataFrame,
    target_col: str = "target",
    reference=None,
) -> lgb.Dataset:
    available = [c for c in ALL_FEATURES if c in df.columns]
    cat_feats = [c for c in CATEGORICAL_FEATURES if c in available]

    # LightGBM requires categorical columns to be pd.Categorical
    df_model = df[available].copy()
    for c in cat_feats:
        df_model[c] = df_model[c].astype("category")

    return lgb.Dataset(
        df_model,
        label=df[target_col].values,
        categorical_feature=cat_feats,
        reference=reference,
        free_raw_data=True,
    )


def train(
    df: pd.DataFrame,
    target_col: str = "target",
    model_path: Path | None = None,
    meta_path: Path | None = None,
    params: dict | None = None,
    valid_frac: float = VALID_FRAC,
    test_frac: float = TEST_FRAC,
    test_seasons: list | None = None,
) -> tuple[lgb.Booster, dict]:
    """
    Train LightGBM on a temporal split. Returns (model, meta).
    meta contains feature list, split dates, train/valid/test AUC.

    If test_seasons is given, those seasons are held out as the test set and the
    remaining data is split temporally into train/valid (1-valid_frac / valid_frac).
    """
    model_path = model_path or MODEL_PATH
    meta_path = meta_path or MODEL_META_PATH
    params = {**DEFAULT_PARAMS, **(params or {})}

    import gc

    # Determine available features before the split (while df is still whole).
    available_features = [c for c in ALL_FEATURES if c in df.columns]
    print(f"  Features: {len(available_features)}")

    # Downcast float64→float32 to halve the numeric footprint before split copies.
    # (~12 GB → ~6 GB for 6.7M×204 cols). LightGBM bins to uint8 internally
    # so float32 precision is sufficient. Skip int columns (some are nullable).
    for col in df.select_dtypes(include="float64").columns:
        df[col] = df[col].astype("float32")
    gc.collect()
    print(f"  DataFrame: {df.memory_usage(deep=False).sum() / 1e9:.1f} GB after downcast")

    if test_seasons:
        # Season-holdout split.  To keep peak memory under control on large DataFrames
        # (12 GB for 6.7M × 204 cols), we avoid extra sort passes:
        #  1. No sort on the copies — df comes pre-sorted from DuckDB queries.
        #  2. del df right after the two boolean-index copies are made, releasing 12 GB.
        test_mask = df["season"].isin(test_seasons)
        test_df   = df.loc[test_mask]    # 2.4 GB copy; peak 12 + 2.4 = 14.4 GB
        remainder = df.loc[~test_mask]   # 8.6 GB copy; peak 12 + 2.4 + 8.6 = 23 GB
        del df; gc.collect()             # free original 12 GB → 11 GB
        n = len(remainder)
        valid_end = int(n * (1 - valid_frac))
        train_df  = remainder.iloc[:valid_end]   # positional view, no copy
        valid_df  = remainder.iloc[valid_end:]   # positional view, no copy
        print(f"  Season-holdout split (test = S{test_seasons}):")
    else:
        train_df, valid_df, test_df = _temporal_split(df, valid_frac, test_frac)
        del df; gc.collect()
        print(f"  Temporal split:")

    print(f"    Train:  {len(train_df):>8,}  ({train_df['started_at'].min()} → {train_df['started_at'].max()})")
    print(f"    Valid:  {len(valid_df):>8,}  ({valid_df['started_at'].min()} → {valid_df['started_at'].max()})")
    print(f"    Test:   {len(test_df):>8,}  ({test_df['started_at'].min()} → {test_df['started_at'].max()})")

    ds_train = _make_dataset(train_df, target_col)
    ds_valid = _make_dataset(valid_df, target_col, reference=ds_train)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ]

    n_est = params.pop("n_estimators", 1000)
    t0 = time.time()
    model = lgb.train(
        params,
        ds_train,
        num_boost_round=n_est,
        valid_sets=[ds_valid],
        callbacks=callbacks,
    )
    print(f"  Training done in {time.time()-t0:.1f}s ({model.num_trees()} trees)")

    from .evaluate import evaluate
    meta = {
        "feature_cols": available_features,
        "cat_features": [c for c in CATEGORICAL_FEATURES if c in available_features],
        "n_trees": model.num_trees(),
        "split": {
            "train_rows": len(train_df),
            "valid_rows": len(valid_df),
            "test_rows": len(test_df),
            "train_end": str(train_df["started_at"].max()),
            "valid_end": str(valid_df["started_at"].max()),
            "test_seasons": test_seasons,
        },
        "metrics": {
            "train": evaluate(train_df[target_col].values, _predict(model, train_df, available_features)),
            "valid": evaluate(valid_df[target_col].values, _predict(model, valid_df, available_features)),
            "test":  evaluate(test_df[target_col].values,  _predict(model, test_df,  available_features)),
        },
        "params": {**params, "n_estimators": n_est},
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"  Model saved → {model_path}")
    return model, meta


def _predict(model: lgb.Booster, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    available = [c for c in feature_cols if c in df.columns]
    cat_feats = [c for c in CATEGORICAL_FEATURES if c in available]
    X = df[available].copy()
    for c in cat_feats:
        X[c] = X[c].astype("category")
    # None values in numeric columns produce object dtype; coerce to float (None → NaN).
    for c in X.select_dtypes(include="object").columns:
        if c not in cat_feats:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return model.predict(X)


def load_model(
    model_path: Path | None = None,
    meta_path: Path | None = None,
) -> tuple[lgb.Booster, dict]:
    model_path = model_path or MODEL_PATH
    meta_path = meta_path or MODEL_META_PATH
    model = lgb.Booster(model_file=str(model_path))
    meta = json.loads(meta_path.read_text())
    return model, meta


def predict_from_features(
    model: lgb.Booster,
    feat_dict: dict,
    feature_cols: list[str],
) -> float:
    """Score a single feature dict. Returns P(player_a wins)."""
    df = pd.DataFrame([feat_dict])
    probs = _predict(model, df, feature_cols)
    return float(probs[0])


# ── XGBoost ───────────────────────────────────────────────────────────────────

XGB_DEFAULT_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 50,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "seed": 42,
    "verbosity": 0,
    "enable_categorical": True,
}


def _make_xgb_dmatrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target",
    global_cats: dict | None = None,
):
    """Build an XGBoost DMatrix, encoding categoricals as pd.Categorical.

    global_cats: {col: list-of-categories} derived from the full dataset before splitting.
    Pass it to every DMatrix creation so train/valid/test share identical category codes,
    preventing XGBoost's 'category not in training set' error at prediction time.
    """
    import xgboost as xgb
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    for c in CATEGORICAL_FEATURES:
        if c in X.columns:
            if global_cats and c in global_cats:
                X[c] = pd.Categorical(X[c], categories=global_cats[c])
            else:
                X[c] = X[c].astype("category")
    label = df[target_col].values if target_col in df.columns else None
    return xgb.DMatrix(X, label=label, enable_categorical=True), available


def train_xgb(
    df: pd.DataFrame,
    target_col: str = "target",
    model_path: Path | None = None,
    meta_path: Path | None = None,
    params: dict | None = None,
    n_rounds: int = 1000,
    early_stopping: int = 50,
    valid_frac: float = VALID_FRAC,
    test_frac: float = TEST_FRAC,
    test_seasons: list | None = None,
) -> tuple:
    """Train XGBoost on a temporal or season-holdout split. Returns (model, meta)."""
    import xgboost as xgb
    model_path = model_path or XGB_MODEL_PATH
    meta_path = meta_path or XGB_META_PATH
    xgb_params = {**XGB_DEFAULT_PARAMS, **(params or {})}

    if test_seasons:
        test_mask = df["season"].isin(test_seasons)
        test_df   = df.loc[test_mask].sort_values("started_at").reset_index(drop=True)
        remainder = df.loc[~test_mask].sort_values("started_at").reset_index(drop=True)
        n = len(remainder)
        valid_end = int(n * (1 - valid_frac))
        train_df  = remainder.iloc[:valid_end]
        valid_df  = remainder.iloc[valid_end:]
        print(f"  Season-holdout split (test = S{test_seasons}):")
    else:
        train_df, valid_df, test_df = _temporal_split(df, valid_frac, test_frac)
        print(f"  Temporal split:")

    print(f"    Train:  {len(train_df):>8,}  ({train_df['started_at'].min()} → {train_df['started_at'].max()})")
    print(f"    Valid:  {len(valid_df):>8,}  ({valid_df['started_at'].min()} → {valid_df['started_at'].max()})")
    print(f"    Test:   {len(test_df):>8,}  ({test_df['started_at'].min()} → {test_df['started_at'].max()})")

    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    print(f"  Features: {len(feature_cols)}")

    # Compute global category vocabulary from the full dataset (train+valid+test combined).
    # This ensures train/valid/test DMatrices share identical category codes, preventing
    # XGBoost's 'category not in training set' error when new patches/civs appear in test.
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    global_cats: dict = {}
    for c in cat_cols:
        vals = sorted(df[c].dropna().unique(), key=str)
        global_cats[c] = [v.item() if hasattr(v, "item") else v for v in vals]

    dm_train, used = _make_xgb_dmatrix(train_df, feature_cols, target_col, global_cats)
    dm_valid, _    = _make_xgb_dmatrix(valid_df, feature_cols, target_col, global_cats)

    t0 = time.time()
    model = xgb.train(
        xgb_params,
        dm_train,
        num_boost_round=n_rounds,
        evals=[(dm_valid, "valid")],
        early_stopping_rounds=early_stopping,
        verbose_eval=100,
    )
    print(f"  Training done in {time.time()-t0:.1f}s ({model.best_iteration} rounds)")

    from .evaluate import evaluate
    meta = {
        "model": "xgboost",
        "feature_cols": used,
        "cat_features": [c for c in CATEGORICAL_FEATURES if c in used],
        "cat_categories": global_cats,
        "n_trees": model.best_iteration,
        "split": {
            "train_rows": len(train_df),
            "valid_rows": len(valid_df),
            "test_rows": len(test_df),
            "train_end": str(train_df["started_at"].max()),
            "valid_end": str(valid_df["started_at"].max()),
            "test_seasons": test_seasons,
        },
        "metrics": {
            "train": evaluate(train_df[target_col].values, _predict_xgb(model, train_df, used, global_cats)),
            "valid": evaluate(valid_df[target_col].values, _predict_xgb(model, valid_df, used, global_cats)),
            "test":  evaluate(test_df[target_col].values,  _predict_xgb(model, test_df,  used, global_cats)),
        },
        "params": xgb_params,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"  Model saved → {model_path}")
    return model, meta


def _predict_xgb(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
    global_cats: dict | None = None,
) -> np.ndarray:
    dm, _ = _make_xgb_dmatrix(df, feature_cols, global_cats=global_cats)
    # Predict at best_iteration (stored by early stopping); fall back to all trees
    best = getattr(model, "best_iteration", None)
    kwargs = {"iteration_range": (0, best + 1)} if best is not None else {}
    return model.predict(dm, **kwargs)


def load_xgb(
    model_path: Path | None = None,
    meta_path: Path | None = None,
) -> tuple:
    import xgboost as xgb
    model_path = model_path or XGB_MODEL_PATH
    meta_path = meta_path or XGB_META_PATH
    model = xgb.Booster()
    model.load_model(str(model_path))
    meta = json.loads(meta_path.read_text())
    return model, meta
