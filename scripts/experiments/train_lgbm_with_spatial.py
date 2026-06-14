"""Retrain LightGBM after merging spatial embeddings with tabular snapshot features.

Usage:
    python scripts/experiments/train_lgbm_with_spatial.py \\
        --base-snapshots data/realtime_outcome_prediction/features/v3_all/snapshots.parquet \\
        --embeddings data/realtime_outcome_prediction/features/v4_spatial/embeddings.parquet \\
        --output-dir data/realtime_outcome_prediction/features/v4_spatial
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss, roc_auc_score

import lightgbm as lgb


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

_META_COLS = {
    "replay_id", "base_replay_id", "snapshot_minute", "snapshot_time_s",
    "snapshot_phase", "latest_event_time_s", "match_duration_observed_s",
    "split", "is_swapped", "target", "row_weight", "snapshots_in_match",
}


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _META_COLS and "_pbgid_" not in c]


def _split(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return df[df["split"] == name].copy()


# ---------------------------------------------------------------------------
# Carry-forward AUC (correct evaluation at clock time T)
# ---------------------------------------------------------------------------

def carry_forward_auc(df: pd.DataFrame, preds: np.ndarray, query_minutes: list[int]) -> dict:
    """For each query time T, take the last available snapshot ≤ T for each match.

    Matches that ended before T are excluded (they have no snapshot at T).
    Returns dict: {"T": {"auc": float, "n": int}}.
    """
    df = df.copy()
    df["pred"] = preds
    results = {}
    for t in query_minutes:
        sub = df[df["snapshot_minute"] <= t].copy()
        if sub.empty:
            continue
        # Keep only the latest snapshot per match
        latest = sub.sort_values("snapshot_minute").groupby("replay_id").last().reset_index()
        if latest["target"].nunique() < 2 or len(latest) < 10:
            continue
        auc = float(roc_auc_score(latest["target"], latest["pred"]))
        results[str(t)] = {"auc": auc, "n": len(latest)}
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="LightGBM + spatial embeddings")
    p.add_argument(
        "--base-snapshots",
        default="data/realtime_outcome_prediction/features/v3_all/snapshots.parquet",
    )
    p.add_argument(
        "--embeddings",
        default="data/realtime_outcome_prediction/features/v4_spatial/embeddings.parquet",
    )
    p.add_argument(
        "--output-dir",
        default="data/realtime_outcome_prediction/features/v4_spatial",
    )
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--min-child-samples", type=int, default=20)
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    args = p.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and merge
    print("Loading snapshots and embeddings...")
    snap = pd.read_parquet(args.base_snapshots)
    emb = pd.read_parquet(args.embeddings)

    emb_cols = [c for c in emb.columns if c.startswith("spatial_")]
    emb1_cols = [c for c in emb_cols if c.startswith("spatial_emb1_")]
    emb2_cols = [c for c in emb_cols if c.startswith("spatial_emb2_")]

    # Embeddings are deduplicated per (replay_id, snapshot_minute); snapshots may have swapped rows.
    # For swapped rows, emb1 and emb2 must be exchanged because slot assignments are reversed.
    merged = snap.merge(
        emb[["replay_id", "snapshot_minute"] + emb_cols],
        on=["replay_id", "snapshot_minute"],
        how="inner",
    )

    if "is_swapped" in merged.columns and emb1_cols and emb2_cols:
        swap_mask = merged["is_swapped"].fillna(False).astype(bool)
        if swap_mask.any():
            tmp1 = merged.loc[swap_mask, emb1_cols].values.copy()
            tmp2 = merged.loc[swap_mask, emb2_cols].values.copy()
            merged.loc[swap_mask, emb1_cols] = tmp2
            merged.loc[swap_mask, emb2_cols] = tmp1
            print(f"  swapped emb1/emb2 for {swap_mask.sum()} rows")

    print(f"  snapshots: {len(snap)}  after merge: {len(merged)}")
    n_emb_cols = sum(1 for c in merged.columns if c.startswith("spatial_"))
    print(f"  spatial embedding columns added: {n_emb_cols}")

    feat_cols = _feature_cols(merged)
    print(f"  total features: {len(feat_cols)}")

    train_df = _split(merged, "train")
    valid_df = _split(merged, "valid")
    test_df = _split(merged, "test")

    X_train = train_df[feat_cols].astype(float)
    y_train = train_df["target"].astype(int)
    w_train = train_df["row_weight"].astype(float) if "row_weight" in train_df else None
    X_valid = valid_df[feat_cols].astype(float)
    y_valid = valid_df["target"].astype(int)
    w_valid = valid_df["row_weight"].astype(float) if "row_weight" in valid_df else None

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "min_child_samples": args.min_child_samples,
        "n_estimators": args.n_estimators,
        "verbose": -1,
    }

    print("Training LightGBM with spatial features...")
    callbacks = [
        lgb.early_stopping(args.early_stopping_rounds, verbose=False),
        lgb.log_evaluation(50),
    ]
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_valid, y_valid)],
        eval_sample_weight=[w_valid] if w_valid is not None else None,
        callbacks=callbacks,
    )

    # Save model
    model_path = output_dir / "lgbm_spatial.txt"
    model.booster_.save_model(str(model_path))
    print(f"Saved model → {model_path}  (best_iter={model.best_iteration_})")

    # Evaluate
    query_minutes = [5, 10, 15, 20, 25, 30]
    results: dict = {}
    for split_name, split_df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        X = split_df[feat_cols].astype(float)
        y = split_df["target"].astype(int)
        preds = model.booster_.predict(X)
        auc = float(roc_auc_score(y, preds))
        ll = float(log_loss(y, preds))
        cf_auc = carry_forward_auc(split_df, preds, query_minutes)
        results[split_name] = {
            "auc": auc,
            "log_loss": ll,
            "n": len(split_df),
            "carry_forward_auc": cf_auc,
        }
        print(
            f"{split_name}: AUC={auc:.4f}  LogLoss={ll:.4f}  n={len(split_df)}"
        )
        for t, m in cf_auc.items():
            print(f"  carry-forward @ {t} min: AUC={m['auc']:.4f}  n={m['n']}")

    meta = {
        "params": params,
        "best_iteration": model.best_iteration_,
        "feature_columns": feat_cols,
        "n_features": len(feat_cols),
        "n_spatial_features": n_emb_cols,
        "results": results,
    }
    meta_path = output_dir / "lgbm_spatial_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved meta → {meta_path}")


if __name__ == "__main__":
    main()
