"""Entry point for the AOE4 rating update investigation.

Usage:
    python -m ratings_delta.run
    python -m ratings_delta.run --db aoe4.duckdb --seasons 10,11
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from the repo root without installing the package
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from aoe4_predict.config import DB_PATH, DEFAULT_TRAIN_SEASONS
from aoe4_predict.db import get_conn

from ratings_delta.dataset import build_dataset, validate_dataset
from ratings_delta.formula import (
    check_determinism,
    fit_elo_grid,
    k_factor_segmentation,
    compute_residuals,
    analyze_residuals,
)
from ratings_delta.baselines import ResultOnlyBaseline, EloBaseline, ResultRatingBucketBaseline
from ratings_delta.model import (
    ALL_FEATURES,
    temporal_split,
    train_lgbm,
    predict,
    compute_metrics,
    compute_subgroup_metrics,
    compute_shap,
)
from ratings_delta.report import generate_report, write_report, REPORT_PATH


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AOE4 visible rating update investigation")
    p.add_argument("--db", default=str(DB_PATH), help="Path to aoe4.duckdb")
    p.add_argument(
        "--seasons",
        default=",".join(str(s) for s in DEFAULT_TRAIN_SEASONS),
        help="Comma-separated season numbers (default: %(default)s)",
    )
    p.add_argument(
        "--no-lgbm",
        action="store_true",
        help="Skip LightGBM training (formula + baseline analysis only)",
    )
    p.add_argument(
        "--no-shap",
        action="store_true",
        help="Skip SHAP computation",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seasons = [int(s.strip()) for s in args.seasons.split(",")]

    print("=" * 60)
    print("AOE4 Visible Rating Update Investigation")
    print(f"  DB:      {args.db}")
    print(f"  Seasons: {seasons}")
    print("=" * 60)

    # ── 1. Data ───────────────────────────────────────────────────
    conn = get_conn(args.db, read_only=True)
    df = build_dataset(conn, seasons)
    conn.close()

    val_stats = validate_dataset(df)

    # ── 2. Formula recovery ───────────────────────────────────────
    det_result = check_determinism(df)
    elo_K, elo_D, elo_mae = fit_elo_grid(df)
    k_segments = k_factor_segmentation(df, elo_D)
    residuals = compute_residuals(df, elo_K, elo_D)
    residual_stats = analyze_residuals(df, residuals)

    # ── 3. Temporal split ─────────────────────────────────────────
    train_df, valid_df, test_df = temporal_split(df)

    # ── 4. Baselines ──────────────────────────────────────────────
    print("\n=== Fitting Baselines ===")

    b_result = ResultOnlyBaseline().fit(train_df)
    b_elo = EloBaseline(K=elo_K, D=elo_D)
    b_bucket = ResultRatingBucketBaseline().fit(train_df)

    print(f"  {b_result}")
    print(f"  {b_elo}")
    print(f"  {b_bucket}")

    target = "observed_rating_delta"
    y_test = test_df[target].values.astype(float)

    pred_result = b_result.predict(test_df)
    pred_elo = b_elo.predict(test_df)
    pred_bucket = b_bucket.predict(test_df)

    # Elo on full dataset for formula quality report
    pred_elo_full = b_elo.predict(df)
    y_full = df[target].values.astype(float)

    all_metrics: dict = {
        "Result-only": compute_metrics(y_test, pred_result),
        "Elo": compute_metrics(y_test, pred_elo),
        "Rating-gap bucket": compute_metrics(y_test, pred_bucket),
        "Elo_full": compute_metrics(y_full, pred_elo_full),
    }

    print("\n  Baseline metrics on test set:")
    for name in ["Result-only", "Elo", "Rating-gap bucket"]:
        m = all_metrics[name]
        print(f"    {name:<22}  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  R²={m['r2']:.4f}")

    # ── 5. LightGBM ───────────────────────────────────────────────
    shap_mean_abs = np.array([])
    feature_names: list = []
    subgroup_metrics: dict = {}

    if not args.no_lgbm:
        model = train_lgbm(train_df, valid_df, target_col=target)
        pred_lgbm = predict(model, test_df)
        lgbm_m = compute_metrics(y_test, pred_lgbm)
        all_metrics["LightGBM"] = lgbm_m
        print(f"\n  LightGBM test: MAE={lgbm_m['mae']:.3f}  RMSE={lgbm_m['rmse']:.3f}  R²={lgbm_m['r2']:.4f}")

        subgroup_metrics = compute_subgroup_metrics(test_df, y_test, pred_lgbm)
        print("\n  Subgroup metrics (LightGBM, test set):")
        for label, m in subgroup_metrics.items():
            print(f"    {label:<35}  MAE={m['mae']:.3f}  R²={m['r2']:.4f}  N={m['n']:,}")

        if not args.no_shap:
            print("\n  Computing SHAP on up to 5,000 test rows...")
            shap_mean_abs, feature_names = compute_shap(model, test_df)
            if len(shap_mean_abs) > 0:
                idx = np.argsort(shap_mean_abs)[::-1][:10]
                print("  Top-10 features by mean |SHAP|:")
                for rank, i in enumerate(idx, 1):
                    print(f"    {rank:2}. {feature_names[i]:<35}  {shap_mean_abs[i]:.4f}")
    else:
        print("\n  Skipping LightGBM (--no-lgbm)")
        feature_names = []

    # ── 6. Report ─────────────────────────────────────────────────
    content = generate_report(
        seasons=seasons,
        val_stats=val_stats,
        det_result=det_result,
        elo_K=elo_K,
        elo_D=elo_D,
        elo_mae=elo_mae,
        k_segments=k_segments,
        residual_stats=residual_stats,
        all_metrics=all_metrics,
        subgroup_metrics=subgroup_metrics,
        shap_mean_abs=shap_mean_abs,
        feature_names=feature_names,
    )
    write_report(content, REPORT_PATH)
    print("\nDone.")


if __name__ == "__main__":
    main()
