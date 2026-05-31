"""
Hyperparameter tuning for LightGBM and XGBoost using Optuna TPE.

Usage:
  python -m aoe4_predict tune --model lgbm --n-trials 50
  python -m aoe4_predict tune --model xgb  --n-trials 50

Datasets are recreated per trial to avoid LightGBM feature_pre_filter state
corruption when min_child_samples varies across trials.

After tuning, re-trains a final model with best params on the full training set
and saves it. Best params are also written to models/{lgbm,xgb}_best_params.json.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MODEL_DIR

# Number of training rows used for each tuning trial.
# Using a fraction keeps each trial fast; the final re-train uses all rows.
TUNE_TRAIN_ROWS = 700_000


def _tune_lgbm(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    n_trials: int = 50,
    timeout: int | None = None,
) -> dict:
    import lightgbm as lgb
    import optuna
    from .model import CATEGORICAL_FEATURES
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cat_feats = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    available = [c for c in feature_cols if c in train_df.columns]

    # Subsample train for speed; valid stays full for stable AUC measurement
    tune_train = (
        train_df.sample(min(TUNE_TRAIN_ROWS, len(train_df)), random_state=42)
        if len(train_df) > TUNE_TRAIN_ROWS else train_df
    )

    def _make_ds(df, ref=None):
        X = df[available].copy()
        for c in cat_feats:
            if c in X.columns:
                X[c] = X[c].astype("category")
        return lgb.Dataset(
            X, label=df["target"].values,
            categorical_feature=cat_feats,
            reference=ref, free_raw_data=True,
        )

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting": "gbdt",
            "verbose": -1,
            "random_state": 42,
            "feature_pre_filter": False,
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": 5,
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
        }
        ds_train = _make_ds(tune_train)
        ds_valid = _make_ds(valid_df, ref=ds_train)
        model = lgb.train(
            params, ds_train,
            num_boost_round=600,
            valid_sets=[ds_valid],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        return model.best_score["valid_0"]["auc"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    return study.best_params


def _tune_xgb(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    n_trials: int = 50,
    timeout: int | None = None,
) -> dict:
    import xgboost as xgb
    import optuna
    from .model import _make_xgb_dmatrix
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    available = [c for c in feature_cols if c in train_df.columns]
    tune_train = (
        train_df.sample(min(TUNE_TRAIN_ROWS, len(train_df)), random_state=42)
        if len(train_df) > TUNE_TRAIN_ROWS else train_df
    )

    # Compute global category vocabulary from train+valid combined so that valid rows
    # with patch/civ values not in the subsample don't cause DMatrix errors.
    from .model import CATEGORICAL_FEATURES
    combined = pd.concat([train_df, valid_df], ignore_index=True)
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in available]
    global_cats = {}
    for c in cat_cols:
        vals = sorted(combined[c].dropna().unique(), key=str)
        global_cats[c] = [v.item() if hasattr(v, "item") else v for v in vals]

    # DMatrix is immutable — create once; valid stays full
    dm_train, _ = _make_xgb_dmatrix(tune_train, available, "target", global_cats)
    dm_valid, _ = _make_xgb_dmatrix(valid_df,   available, "target", global_cats)

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "verbosity": 0,
            "seed": 42,
            "enable_categorical": True,
            "eta": trial.suggest_float("eta", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 10, 200),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "max_leaves": trial.suggest_int("max_leaves", 0, 255),
        }
        model = xgb.train(
            params, dm_train,
            num_boost_round=600,
            evals=[(dm_valid, "valid")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        return model.best_score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    return study.best_params


def run_tune(
    df: pd.DataFrame,
    model_type: str = "lgbm",
    n_trials: int = 50,
    timeout: int | None = None,
    retrain: bool = True,
    valid_frac: float = 0.15,
    test_frac: float = 0.15,
    test_seasons: list | None = None,
    model_path: Path | None = None,
    meta_path: Path | None = None,
) -> dict:
    """
    Run Optuna hyperparameter search on a temporal or season-holdout split.
    Trials use TUNE_TRAIN_ROWS subsampled rows for speed.
    Final re-train (if retrain=True) uses the full training split.
    Returns best_params dict.
    """
    from .model import ALL_FEATURES

    if test_seasons:
        test_mask = df["season"].isin(test_seasons)
        remainder = df[~test_mask].sort_values("started_at").reset_index(drop=True)
        n = len(remainder)
        valid_end = int(n * (1 - valid_frac))
        train_df = remainder.iloc[:valid_end].copy()
        valid_df = remainder.iloc[valid_end:].copy()
    else:
        df_sorted = df.sort_values("started_at").reset_index(drop=True)
        n = len(df_sorted)
        train_end = int(n * (1 - valid_frac - test_frac))
        valid_end = int(n * (1 - test_frac))
        train_df = df_sorted.iloc[:train_end].copy()
        valid_df = df_sorted.iloc[train_end:valid_end].copy()

    feature_cols = [c for c in ALL_FEATURES if c in df.columns]

    print(f"Tuning {model_type.upper()} — {n_trials} trials")
    print(f"  Train: {len(train_df):,} rows (trials use ≤{TUNE_TRAIN_ROWS:,}), "
          f"Valid: {len(valid_df):,} rows, Features: {len(feature_cols)}")

    t0 = time.time()
    if model_type == "lgbm":
        best_params = _tune_lgbm(train_df, valid_df, feature_cols, n_trials, timeout)
    elif model_type == "xgb":
        best_params = _tune_xgb(train_df, valid_df, feature_cols, n_trials, timeout)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r} (must be 'lgbm' or 'xgb')")

    elapsed = time.time() - t0
    print(f"\nTuning complete in {elapsed:.0f}s  (best valid AUC from last trial)")
    print("Best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Persist best params
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    params_path = MODEL_DIR / f"{model_type}_best_params.json"
    params_path.write_text(json.dumps(best_params, indent=2))
    print(f"Best params saved → {params_path}")

    if retrain:
        print(f"\nRe-training {model_type.upper()} on full training split with best params...")
        if model_type == "lgbm":
            from .model import train, DEFAULT_PARAMS
            final_params = {**DEFAULT_PARAMS, **best_params, "feature_pre_filter": False}
            _, meta = train(df, params=final_params, test_seasons=test_seasons,
                            model_path=model_path, meta_path=meta_path)
        else:
            from .model import train_xgb, XGB_DEFAULT_PARAMS
            final_params = {**XGB_DEFAULT_PARAMS, **best_params}
            _, meta = train_xgb(df, params=final_params, test_seasons=test_seasons,
                                model_path=model_path, meta_path=meta_path)

        print(f"\n── {model_type.upper()} Tuned Metrics ──")
        for split, m in meta["metrics"].items():
            print(f"  {split:<6}  AUC={m['auc']:.4f}  LogLoss={m['log_loss']:.4f}  Brier={m['brier']:.4f}")

    return best_params
