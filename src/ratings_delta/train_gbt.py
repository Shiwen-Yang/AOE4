"""Train and save the deployment GBT (LightGBM) rating-delta model.

Rebuilds the participant-level dataset from the DB (the saved CSV splits only
carry the parametric model's columns), trains with the standard temporal
split, evaluates on the test slice, and saves the booster + metadata to
models/ratings_delta/lgbm_delta.txt / lgbm_delta_meta.json — the artifacts the
backend serves at /predict/rating-delta.

Usage (from repo root):
    PYTHONPATH=src python -m ratings_delta.train_gbt
    PYTHONPATH=src python -m ratings_delta.train_gbt --db aoe4.duckdb --seasons 10,11,12
"""
import argparse
import json
from datetime import datetime, timezone


def parse_args() -> argparse.Namespace:
    from aoe4_predict.config import DB_PATH, DEFAULT_TRAIN_SEASONS

    p = argparse.ArgumentParser(description="Train deployment GBT rating-delta model")
    p.add_argument("--db", default=str(DB_PATH), help="Path to aoe4.duckdb")
    p.add_argument(
        "--seasons",
        default=",".join(str(s) for s in DEFAULT_TRAIN_SEASONS),
        help="Comma-separated season numbers (default: %(default)s)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np

    from aoe4_predict.config import BASE_DIR
    from aoe4_predict.db import get_conn
    from ratings_delta.dataset import build_dataset, validate_dataset
    from ratings_delta.model import (
        ALL_FEATURES,
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
        compute_metrics,
        predict,
        save_lgbm,
        temporal_split,
        train_lgbm,
    )

    seasons = [int(s.strip()) for s in args.seasons.split(",")]
    model_dir = BASE_DIR / "models" / "ratings_delta"
    model_path = model_dir / "lgbm_delta.txt"
    meta_path = model_dir / "lgbm_delta_meta.json"

    print("=" * 60)
    print("Deployment GBT rating-delta model")
    print(f"  DB:      {args.db}")
    print(f"  Seasons: {seasons}")
    print("=" * 60)

    conn = get_conn(args.db, read_only=True)
    df = build_dataset(conn, seasons)
    conn.close()
    validate_dataset(df)

    train_df, valid_df, test_df = temporal_split(df)
    model = train_lgbm(train_df, valid_df)

    target = "observed_rating_delta"
    y_test = test_df[target].values.astype(float)
    pred_test = predict(model, test_df)
    m_raw = compute_metrics(y_test, pred_test)
    # Deployment applies the regular 0.5-threshold rounding rule.
    pred_rounded = np.sign(pred_test) * np.floor(np.abs(pred_test) + 0.5)
    m_rounded = compute_metrics(y_test, pred_rounded)

    print(f"\n  Test (raw):     MAE={m_raw['mae']:.4f}  RMSE={m_raw['rmse']:.4f}  R²={m_raw['r2']:.4f}")
    print(f"  Test (rounded): MAE={m_rounded['mae']:.4f}  RMSE={m_rounded['rmse']:.4f}  R²={m_rounded['r2']:.4f}")

    meta = {
        "model_type": "lightgbm_regression",
        "target": target,
        "seasons": seasons,
        "feature_cols": ALL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "cat_features": CATEGORICAL_FEATURES,
        "best_iteration": model.best_iteration_,
        "rounding": "half_away_from_zero",
        "metrics": {"test_raw": m_raw, "test_rounded": m_rounded},
        "split": {
            "train_rows": len(train_df),
            "valid_rows": len(valid_df),
            "test_rows": len(test_df),
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    save_lgbm(model, model_path, meta, meta_path)
    print(f"\n  Saved model → {model_path}")
    print(f"  Saved meta  → {meta_path}")

    # round-trip verify
    from ratings_delta.model import load_lgbm, predict_booster

    booster, loaded_meta = load_lgbm(model_path, meta_path)
    pred_loaded = predict_booster(booster, test_df.head(1000))
    assert np.allclose(pred_loaded, pred_test[:1000], atol=1e-9), "Load/save round-trip failed"
    assert loaded_meta["best_iteration"] == model.best_iteration_
    print("  Round-trip verified.")
    print(json.dumps({"test_raw_mae": m_raw["mae"], "test_rounded_mae": m_rounded["mae"]}))


if __name__ == "__main__":
    main()
