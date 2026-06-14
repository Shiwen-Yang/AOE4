from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Non-feature columns excluded from the training matrix
_META_COLS = {
    "replay_id",
    "base_replay_id",
    "snapshot_minute",
    "snapshot_time_s",
    "snapshot_phase",
    "latest_event_time_s",
    "match_duration_observed_s",
    "split",
    "is_swapped",
    "target",
    "row_weight",
    "snapshots_in_match",
}


def feature_columns(df: pd.DataFrame) -> list[str]:
    # Exclude per-entity pbgid columns: too sparse (32K+ cols, most zero per row)
    return [c for c in df.columns if c not in _META_COLS and "_pbgid_" not in c]


def _split(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return df[df["split"] == name].copy()


def train_lgbm(
    snapshots_path: Path,
    output_dir: Path,
    num_leaves: int = 63,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    min_child_samples: int = 20,
    early_stopping_rounds: int = 50,
) -> dict[str, Any]:
    import lightgbm as lgb

    df = pd.read_parquet(snapshots_path)
    train = _split(df, "train")
    valid = _split(df, "valid")

    if train.empty:
        raise ValueError("No training rows — run build-dataset first")

    feat_cols = feature_columns(df)
    X_train = train[feat_cols].astype(float)
    y_train = train["target"].astype(int)
    w_train = train["row_weight"].astype(float) if "row_weight" in train else None

    X_valid = valid[feat_cols].astype(float)
    y_valid = valid["target"].astype(int)
    w_valid = valid["row_weight"].astype(float) if "row_weight" in valid else None

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "min_child_samples": min_child_samples,
        "n_estimators": n_estimators,
        "verbose": -1,
    }

    callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(50)]

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_valid, y_valid)],
        eval_sample_weight=[w_valid] if w_valid is not None else None,
        callbacks=callbacks,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "lgbm_realtime.txt"
    model.booster_.save_model(str(model_path))

    meta = {
        "feature_columns": feat_cols,
        "n_features": len(feat_cols),
        "best_iteration": model.best_iteration_,
        "train_rows": len(train),
        "valid_rows": len(valid),
        "params": params,
    }
    (output_dir / "lgbm_realtime_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Saved model to {model_path} (best_iteration={model.best_iteration_})")
    return meta


def load_lgbm(output_dir: Path):
    import lightgbm as lgb

    model_path = output_dir / "lgbm_realtime.txt"
    meta_path = output_dir / "lgbm_realtime_meta.json"
    booster = lgb.Booster(model_file=str(model_path))
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return booster, meta


def evaluate(
    snapshots_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    from sklearn.metrics import log_loss, roc_auc_score

    booster, meta = load_lgbm(output_dir)
    feat_cols = meta.get("feature_columns") or []

    df = pd.read_parquet(snapshots_path)
    results: dict[str, Any] = {}

    for split_name in ("train", "valid", "test"):
        split_df = _split(df, split_name)
        if split_df.empty or "target" not in split_df:
            continue

        missing = [c for c in feat_cols if c not in split_df.columns]
        if missing:
            print(f"Warning: {len(missing)} feature columns missing in {split_name} split")

        X = split_df[[c for c in feat_cols if c in split_df.columns]].astype(float)
        y = split_df["target"].astype(int)
        preds = booster.predict(X)

        auc = float(roc_auc_score(y, preds))
        loss = float(log_loss(y, preds))

        # Per-phase breakdown
        phase_metrics: dict[str, dict] = {}
        for phase in split_df["snapshot_phase"].unique():
            mask = split_df["snapshot_phase"] == phase
            if mask.sum() < 10:
                continue
            phase_metrics[phase] = {
                "auc": float(roc_auc_score(y[mask], preds[mask])),
                "n": int(mask.sum()),
            }

        # Per-minute breakdown
        minute_metrics: dict[str, dict] = {}
        for minute in sorted(split_df["snapshot_minute"].unique()):
            mask = split_df["snapshot_minute"] == minute
            if mask.sum() < 10:
                continue
            minute_metrics[str(int(minute))] = {
                "auc": float(roc_auc_score(y[mask], preds[mask])),
                "n": int(mask.sum()),
            }

        results[split_name] = {
            "auc": auc,
            "log_loss": loss,
            "n": len(split_df),
            "by_phase": phase_metrics,
            "by_minute": minute_metrics,
        }
        print(f"{split_name}: AUC={auc:.4f}  LogLoss={loss:.4f}  n={len(split_df)}")

    report_path = output_dir / "eval_report.json"
    report_path.write_text(json.dumps(results, indent=2))
    return results
