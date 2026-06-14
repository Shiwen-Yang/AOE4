"""Entry point for the AOE4 rating update investigation.

Usage:
    python -m ratings_delta.run
    python -m ratings_delta.run --db aoe4.duckdb --seasons 10,11
"""
import argparse


def parse_args() -> argparse.Namespace:
    from aoe4_predict.config import DB_PATH, DEFAULT_TRAIN_SEASONS

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

    import numpy as np

    from aoe4_predict.db import get_conn
    from ratings_delta.dataset import build_dataset, validate_dataset
    from ratings_delta.formula import (
        check_determinism,
        fit_elo_grid,
        k_factor_segmentation,
        compute_residuals,
        analyze_residuals,
    )
    from ratings_delta.baselines import ResultOnlyBaseline, EloBaseline, ResultRatingBucketBaseline, DynamicKEloBaseline, PiecewiseInterceptEloBaseline
    from ratings_delta.model import (
        temporal_split,
        train_lgbm,
        predict,
        compute_metrics,
        compute_subgroup_metrics,
        compute_shap,
    )
    from ratings_delta.report import generate_report, write_report, REPORT_PATH

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
    b_dyn = DynamicKEloBaseline().fit(train_df)
    b_pw  = PiecewiseInterceptEloBaseline(kd_baseline=b_dyn).fit(train_df)

    print(f"  {b_result}")
    print(f"  {b_elo}")
    print(f"  {b_bucket}")
    print(f"  {b_dyn}")
    print()
    print(b_pw.report())

    target = "observed_rating_delta"
    y_test = test_df[target].values.astype(float)

    pred_result = b_result.predict(test_df)
    pred_elo = b_elo.predict(test_df)
    pred_bucket = b_bucket.predict(test_df)
    pred_dyn = b_dyn.predict(test_df)
    pred_pw  = b_pw.predict(test_df)

    # Elo on full dataset for formula quality report
    pred_elo_full = b_elo.predict(df)
    y_full = df[target].values.astype(float)

    all_metrics: dict = {
        "Result-only": compute_metrics(y_test, pred_result),
        "Elo": compute_metrics(y_test, pred_elo),
        "Rating-gap bucket": compute_metrics(y_test, pred_bucket),
        "Dynamic-K Elo": compute_metrics(y_test, pred_dyn),
        "Piecewise Elo": compute_metrics(y_test, pred_pw),
        "Elo_full": compute_metrics(y_full, pred_elo_full),
    }

    print("\n  Baseline metrics on test set:")
    for name in ["Result-only", "Elo", "Rating-gap bucket", "Dynamic-K Elo", "Piecewise Elo"]:
        m = all_metrics[name]
        print(f"    {name:<22}  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  R²={m['r2']:.4f}")

    # ── 5. LightGBM on raw target ─────────────────────────────────
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

    # ── 6. LightGBM on Dynamic-K Elo residuals ───────────────────
    residual_shap_mean_abs = np.array([])
    residual_feature_names: list = []

    if not args.no_lgbm:
        residual_col = "dyn_elo_residual"
        train_df[residual_col] = b_pw.residuals(train_df)
        valid_df[residual_col] = b_pw.residuals(valid_df)
        test_df[residual_col]  = b_pw.residuals(test_df)

        residual_model = train_lgbm(train_df, valid_df, target_col=residual_col)
        pred_residual = predict(residual_model, test_df)
        y_residual_test = test_df[residual_col].values.astype(float)
        res_m = compute_metrics(y_residual_test, pred_residual)
        all_metrics["LightGBM (residual)"] = res_m
        print(f"\n  Residual LightGBM test: MAE={res_m['mae']:.3f}  RMSE={res_m['rmse']:.3f}  R²={res_m['r2']:.4f}")

        # Piecewise Elo + residual model combined
        pred_combined = pred_pw + pred_residual
        combined_m = compute_metrics(y_test, pred_combined)
        all_metrics["Piecewise Elo + residual LGB"] = combined_m
        print(f"  Combined (Piecewise Elo + residual) test: MAE={combined_m['mae']:.3f}  RMSE={combined_m['rmse']:.3f}  R²={combined_m['r2']:.4f}")

        if not args.no_shap:
            print("\n  Computing SHAP on residual model (up to 5,000 rows)...")
            residual_shap_mean_abs, residual_feature_names = compute_shap(residual_model, test_df)
            if len(residual_shap_mean_abs) > 0:
                idx = np.argsort(residual_shap_mean_abs)[::-1][:10]
                print("  Top-10 features explaining Dynamic-K Elo residual:")
                for rank, i in enumerate(idx, 1):
                    print(f"    {rank:2}. {residual_feature_names[i]:<35}  {residual_shap_mean_abs[i]:.4f}")
    else:
        print("\n  Skipping LightGBM (--no-lgbm)")
        feature_names = []

    # ── 6. Piecewise intercept plot ───────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        # raw residual = obs - K*D_pred (no intercept)
        # = b_pw.residuals(test_df) + intercept(games)
        g_test = test_df["games_this_season_before"].fillna(-1).values
        intercept_test = b_pw._intercept(g_test)
        raw_resid = b_pw.residuals(test_df) + intercept_test

        cap = 200
        tmp = pd.DataFrame({"g": g_test.astype(int), "r": raw_resid})
        tmp = tmp[tmp["g"].between(0, cap)]
        agg = tmp.groupby("g")["r"].agg(["mean", "count"]).reset_index()
        agg.columns = ["g", "mean_r", "n"]
        agg["se"] = (agg["mean_r"].std() / np.sqrt(agg["n"])).fillna(0)

        g_fit = np.arange(0, cap + 1, dtype=float)
        b_fit = b_pw._intercept(g_fit)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.fill_between(agg["g"], agg["mean_r"] - 1.96 * agg["se"],
                        agg["mean_r"] + 1.96 * agg["se"], alpha=0.2, color="steelblue")
        ax.scatter(agg["g"], agg["mean_r"], s=4, color="steelblue", alpha=0.6, label="Per-game mean residual")
        ax.plot(g_fit, b_fit, color="crimson", linewidth=2.0, label="Piecewise fit")
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        for cp in b_pw.changepoints:
            ax.axvline(cp, color="grey", linewidth=0.9, linestyle=":", alpha=0.8)
        ax.set_xlabel("games_this_season_before")
        ax.set_ylabel("Mean residual (observed − KD·Elo)")
        ax.set_title("Piecewise intercept b(n) vs per-game mean residual (test set)")
        ax.legend()
        ax.set_xlim(0, cap)
        fig.tight_layout()
        from aoe4_predict.config import FIGURES_DIR
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIGURES_DIR / "piecewise_intercept_fit.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"\n  Plot saved to {out_path}")
    except Exception as e:
        print(f"\n  (Plot skipped: {e})")

    # ── 7. Report ─────────────────────────────────────────────────
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
