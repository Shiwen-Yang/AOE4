"""Entry point for civ-choice prediction.

Usage:
    python3 -m civ_choice.run
    python3 -m civ_choice.run --db aoe4.duckdb --seasons 10,11
    python3 -m civ_choice.run --no-lgbm
    python3 -m civ_choice.run --no-shap
"""
import argparse


def parse_args() -> argparse.Namespace:
    from aoe4_predict.config import DB_PATH, DEFAULT_TRAIN_SEASONS

    p = argparse.ArgumentParser(description="AOE4 civ-choice prediction")
    p.add_argument("--db", default=str(DB_PATH))
    p.add_argument(
        "--seasons",
        default=",".join(str(s) for s in DEFAULT_TRAIN_SEASONS),
        help="Comma-separated season numbers",
    )
    p.add_argument("--no-lgbm", action="store_true", help="Skip LightGBM training")
    p.add_argument("--no-shap", action="store_true", help="Skip SHAP computation")
    p.add_argument(
        "--rebuild", action="store_true",
        help="Rebuild DuckDB tables even if they already exist"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np

    from aoe4_predict.db import get_conn, table_exists
    from civ_choice.dataset import build_tables, load_training_matrix, validate_dataset
    from civ_choice.features import add_derived_features
    from civ_choice.baselines import (
        GlobalPickRateBaseline,
        LifetimeFreqBaseline,
        Recent30dBaseline,
        MapSpecificBaseline,
        LastCivBaseline,
        normalize_within_group,
    )
    from civ_choice.model import (
        temporal_split,
        train_lgbm,
        save_model,
        normalize_predictions,
        normalize_raw_scores_with_temperature,
        fit_temperature,
        compute_group_metrics,
        compute_subgroup_metrics,
        compute_shap,
    )
    from civ_choice.report import generate_report, write_report, REPORT_PATH

    seasons = [int(s.strip()) for s in args.seasons.split(",")]

    print("=" * 60)
    print("AOE4 Civilization-Choice Prediction (V1)")
    print(f"  DB:      {args.db}")
    print(f"  Seasons: {seasons}")
    print("=" * 60)

    # ── 1. Build / load dataset ───────────────────────────────────────────
    conn = get_conn(args.db)

    if args.rebuild or not table_exists(conn, "civ_choice_training_matrix"):
        build_tables(conn, seasons)
    else:
        print("  civ_choice_training_matrix already exists — skipping rebuild (use --rebuild to force)")

    # Get randomized pct for report
    seasons_str = ", ".join(str(s) for s in seasons)
    total_rows = conn.execute(
        f"SELECT COUNT(*) FROM participants p JOIN games g ON p.game_id = g.game_id"
        f" WHERE g.kind IN ('rm_1v1','rm_solo') AND p.result IS NOT NULL"
        f" AND g.season IN ({seasons_str})"
    ).fetchone()[0]
    rand_rows = conn.execute(
        f"SELECT COUNT(*) FROM participants p JOIN games g ON p.game_id = g.game_id"
        f" WHERE g.kind IN ('rm_1v1','rm_solo') AND p.result IS NOT NULL"
        f" AND g.season IN ({seasons_str})"
        f" AND (p.civilization IS NULL OR p.civilization_randomized = TRUE)"
    ).fetchone()[0]
    randomized_pct = rand_rows / total_rows * 100 if total_rows else 0

    df_raw = load_training_matrix(conn, pioneer=True)
    conn.close()

    val_stats = validate_dataset(df_raw)

    # ── 2. Derived features ───────────────────────────────────────────────
    print("\n  Adding derived features...")
    df = add_derived_features(df_raw)

    # ── 3. Temporal split ─────────────────────────────────────────────────
    train_df, valid_df, test_df = temporal_split(df)

    target_col = "target"
    y_test = test_df[target_col].values

    # ── 4. Baselines ──────────────────────────────────────────────────────
    print("\n=== Fitting Baselines ===")
    baselines = {
        "Global pick-rate": GlobalPickRateBaseline(),
        "Lifetime freq":    LifetimeFreqBaseline(),
        "Recent 30d":       Recent30dBaseline(),
        "Map-specific":     MapSpecificBaseline(),
        "Last-civ":         LastCivBaseline(),
    }
    all_metrics: dict = {}
    for name, bl in baselines.items():
        bl.fit(train_df)
        raw = bl.predict_proba(test_df)
        norm = normalize_within_group(test_df, raw)
        m = compute_group_metrics(test_df, norm)
        all_metrics[name] = m
        print(f"  {name:<22}  Top1={m.get('top1_acc', 0):.3f}  LogLoss={m.get('log_loss', 99):.4f}")

    # ── 5. LightGBM ───────────────────────────────────────────────────────
    shap_mean = np.array([])
    feature_names: list = []
    subgroup_metrics: dict = {}
    normalization_winner = "renorm"

    if not args.no_lgbm:
        model = train_lgbm(train_df, valid_df)

        # Compare renorm vs softmax on validation set
        from civ_choice.features import prepare_X
        X_valid = prepare_X(valid_df)
        raw_val = model.predict_proba(X_valid)[:, 1]
        raw_score_val = model.predict(X_valid, raw_score=True)
        norm_renorm = normalize_predictions(valid_df, raw_val, method="renorm")
        norm_softmax = normalize_predictions(valid_df, raw_val, method="softmax")
        temperature, temp_valid_nll = fit_temperature(valid_df, raw_score_val)
        norm_temp = normalize_raw_scores_with_temperature(valid_df, raw_score_val, temperature)
        m_renorm = compute_group_metrics(valid_df, norm_renorm)
        m_softmax = compute_group_metrics(valid_df, norm_softmax)
        m_temp = compute_group_metrics(valid_df, norm_temp)
        normalization_options = {
            "renorm": m_renorm,
            "softmax": m_softmax,
            "temperature": m_temp,
        }
        normalization_winner = min(
            normalization_options,
            key=lambda name: normalization_options[name].get("log_loss", 99),
        )
        print(f"\n  Normalization: renorm LogLoss={m_renorm.get('log_loss', 99):.4f}  "
              f"softmax LogLoss={m_softmax.get('log_loss', 99):.4f}  "
              f"temperature(T={temperature:.3f}) LogLoss={temp_valid_nll:.4f}  "
              f"→ using {normalization_winner}")

        X_test = prepare_X(test_df)
        raw_test = model.predict_proba(X_test)[:, 1]
        if normalization_winner == "temperature":
            raw_score_test = model.predict(X_test, raw_score=True)
            norm_test = normalize_raw_scores_with_temperature(test_df, raw_score_test, temperature)
        else:
            norm_test = normalize_predictions(test_df, raw_test, method=normalization_winner)
        lgbm_m = compute_group_metrics(test_df, norm_test)
        all_metrics["LightGBM"] = lgbm_m
        print(f"\n  LightGBM test:  Top1={lgbm_m.get('top1_acc', 0):.3f}  "
              f"Top3={lgbm_m.get('top3_acc', 0):.3f}  "
              f"LogLoss={lgbm_m.get('log_loss', 99):.4f}")
        save_model(
            model,
            meta={
                "seasons": seasons,
                "normalization": normalization_winner,
                "temperature": temperature if normalization_winner == "temperature" else None,
                "metrics": {"test": lgbm_m, "validation_normalization": normalization_options},
            },
        )
        print("  Saved LightGBM model to models/civ_choice/lgbm_civ_choice.txt")

        subgroup_metrics = compute_subgroup_metrics(test_df, norm_test)
        print("\n  Subgroup metrics (test set):")
        for label, m in subgroup_metrics.items():
            if m:
                print(f"    {label:<35}  Top1={m['top1_acc']:.3f}  N={m['n']:,}")

        if not args.no_shap:
            print("\n  Computing SHAP on up to 5,000 test rows...")
            shap_mean, feature_names = compute_shap(model, test_df)
            if len(shap_mean) > 0:
                idx = np.argsort(shap_mean)[::-1][:10]
                print("  Top-10 features by mean |SHAP|:")
                for rank, i in enumerate(idx, 1):
                    print(f"    {rank:2}. {feature_names[i]:<40}  {shap_mean[i]:.4f}")
    else:
        print("\n  Skipping LightGBM (--no-lgbm)")

    # ── 6. Report ─────────────────────────────────────────────────────────
    content = generate_report(
        seasons=seasons,
        val_stats=val_stats,
        randomized_pct=randomized_pct,
        all_metrics=all_metrics,
        subgroup_metrics=subgroup_metrics,
        shap_mean=shap_mean,
        feature_names=feature_names,
        normalization_winner=normalization_winner,
    )
    write_report(content, REPORT_PATH)
    print("\nDone.")


if __name__ == "__main__":
    main()
