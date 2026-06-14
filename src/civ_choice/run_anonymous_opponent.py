"""Entry point for anonymous opponent civ-choice prediction.

Usage:
    python3 -m civ_choice.run_anonymous_opponent
    python3 -m civ_choice.run_anonymous_opponent --db aoe4.duckdb --seasons 10,11
    python3 -m civ_choice.run_anonymous_opponent --user-profile-id 123456
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    from aoe4_predict.config import DB_PATH, DEFAULT_TRAIN_SEASONS

    p = argparse.ArgumentParser(
        description="Live-assistant civ-choice prediction for a known user's opponents"
    )
    p.add_argument("--db", default=str(DB_PATH))
    p.add_argument(
        "--seasons",
        default=",".join(str(s) for s in DEFAULT_TRAIN_SEASONS),
        help="Comma-separated season numbers",
    )
    p.add_argument(
        "--user-profile-id",
        type=int,
        default=None,
        help="Known user profile. Predict civs for opponents in this user's games.",
    )
    p.add_argument("--no-lgbm", action="store_true", help="Skip LightGBM training")
    p.add_argument("--no-shap", action="store_true", help="Skip SHAP computation")
    p.add_argument("--rebuild", action="store_true", help="Rebuild DuckDB tables")
    p.add_argument(
        "--sample-mod",
        type=int,
        default=10,
        help="Generic mode samples games where hash(game_id) %% sample_mod < sample_keep.",
    )
    p.add_argument(
        "--sample-keep",
        type=int,
        default=2,
        help="Generic mode sample numerator.",
    )
    return p.parse_args()


def _table_matches_mode(conn, requested_user_profile_id: int | None) -> bool:
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS n,
            COUNT(*) FILTER (WHERE user_profile_id IS NOT NULL) AS n_user,
            MIN(user_profile_id) AS min_user,
            MAX(user_profile_id) AS max_user
        FROM anonymous_opponent_training_matrix
        """
    ).fetchone()
    n, n_user, min_user, max_user = row
    if n == 0:
        return False
    if requested_user_profile_id is None:
        return n_user == 0
    return n_user == n and min_user == requested_user_profile_id and max_user == requested_user_profile_id


def _anonymous_subgroup_metrics(df, y_pred_norm):
    from civ_choice.model import compute_group_metrics

    tmp = df.copy()
    tmp["_pred"] = y_pred_norm
    masks = {
        "MMR low": tmp["mmr_bucket"].astype(str) == "low",
        "MMR mid": tmp["mmr_bucket"].astype(str) == "mid",
        "MMR high": tmp["mmr_bucket"].astype(str) == "high",
        "MMR unknown": tmp["mmr_bucket"].astype(str) == "unknown",
    }
    top_maps = (
        tmp[tmp["target"] == 1]["map"].astype(str).value_counts().head(5).index.tolist()
    )
    for map_name in top_maps:
        masks[f"Map: {map_name}"] = tmp["map"].astype(str) == map_name

    results = {}
    for label, mask in masks.items():
        sub = tmp[mask]
        if sub.groupby(["game_id", "profile_id"]).ngroups < 25:
            continue
        results[label] = compute_group_metrics(sub, sub["_pred"].values)
    return results


def _compute_anonymous_shap(model, df, max_rows: int = 5000):
    import numpy as np

    from civ_choice.anonymous_features import ANONYMOUS_FEATURES, prepare_anonymous_X

    try:
        import shap
    except ImportError:
        print("  shap not installed — skipping SHAP analysis")
        return np.array([]), ANONYMOUS_FEATURES

    X = prepare_anonymous_X(df).sample(min(max_rows, len(df)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    return np.abs(shap_vals).mean(axis=0), ANONYMOUS_FEATURES


def main() -> None:
    args = parse_args()

    import numpy as np

    from aoe4_predict.db import get_conn, table_exists
    from civ_choice.anonymous_baselines import (
        AnonymousGlobalPickRateBaseline,
        MMRTierPickRateBaseline,
        MapPatchPickRateBaseline,
        RecentOpponentMetaBaseline,
    )
    from civ_choice.anonymous_dataset import (
        build_anonymous_tables,
        load_anonymous_training_matrix,
        validate_anonymous_dataset,
    )
    from civ_choice.anonymous_features import add_anonymous_derived_features, prepare_anonymous_X
    from civ_choice.anonymous_model import train_anonymous_lgbm
    from civ_choice.anonymous_report import (
        REPORT_PATH,
        generate_anonymous_report,
        write_report,
    )
    from civ_choice.baselines import normalize_within_group
    from civ_choice.model import (
        compute_group_metrics,
        fit_temperature,
        normalize_predictions,
        normalize_raw_scores_with_temperature,
        temporal_split,
    )

    seasons = [int(s.strip()) for s in args.seasons.split(",")]

    print("=" * 68)
    print("AOE4 Anonymous Opponent Civilization-Choice Prediction")
    print(f"  DB:              {args.db}")
    print(f"  Seasons:         {seasons}")
    print(f"  User profile ID: {args.user_profile_id}")
    print("=" * 68)

    conn = get_conn(args.db)
    has_table = table_exists(conn, "anonymous_opponent_training_matrix")
    reuse_table = (
        has_table
        and not args.rebuild
        and _table_matches_mode(conn, args.user_profile_id)
    )
    if reuse_table:
        print("  anonymous_opponent_training_matrix already matches requested mode — skipping rebuild")
    else:
        build_anonymous_tables(
            conn,
            seasons,
            user_profile_id=args.user_profile_id,
            sample_mod=args.sample_mod,
            sample_keep=args.sample_keep,
        )

    df_raw = load_anonymous_training_matrix(conn)
    conn.close()

    val_stats = validate_anonymous_dataset(df_raw)
    print("\n  Adding anonymous derived features...")
    df = add_anonymous_derived_features(df_raw)
    train_df, valid_df, test_df = temporal_split(df)

    print("\n=== Fitting Anonymous Baselines ===")
    baselines = {
        "Global pick-rate": AnonymousGlobalPickRateBaseline(),
        "MMR-tier pick-rate": MMRTierPickRateBaseline(),
        "Map-patch pick-rate": MapPatchPickRateBaseline(),
        "Recent-opponent meta": RecentOpponentMetaBaseline(),
    }
    all_metrics: dict[str, dict] = {}
    for name, baseline in baselines.items():
        baseline.fit(train_df)
        raw = baseline.predict_proba(test_df)
        norm = normalize_within_group(test_df, raw)
        metrics = compute_group_metrics(test_df, norm)
        all_metrics[name] = metrics
        print(
            f"  {name:<24} Top1={metrics.get('top1_acc', 0):.3f} "
            f"LogLoss={metrics.get('log_loss', 99):.4f}"
        )

    shap_mean = np.array([])
    feature_names: list[str] = []
    subgroup_metrics: dict[str, dict] = {}
    normalization_winner = "baseline-only"

    if not args.no_lgbm:
        model = train_anonymous_lgbm(train_df, valid_df)

        X_valid = prepare_anonymous_X(valid_df)
        raw_val = model.predict_proba(X_valid)[:, 1]
        raw_score_val = model.predict(X_valid, raw_score=True)
        norm_renorm = normalize_predictions(valid_df, raw_val, method="renorm")
        norm_softmax = normalize_predictions(valid_df, raw_val, method="softmax")
        temperature, temp_valid_nll = fit_temperature(valid_df, raw_score_val)
        norm_temp = normalize_raw_scores_with_temperature(valid_df, raw_score_val, temperature)
        normalization_options = {
            "renorm": compute_group_metrics(valid_df, norm_renorm),
            "softmax": compute_group_metrics(valid_df, norm_softmax),
            "temperature": compute_group_metrics(valid_df, norm_temp),
        }
        normalization_winner = min(
            normalization_options,
            key=lambda name: normalization_options[name].get("log_loss", 99),
        )
        print(
            f"\n  Normalization: renorm LogLoss={normalization_options['renorm'].get('log_loss', 99):.4f}  "
            f"softmax LogLoss={normalization_options['softmax'].get('log_loss', 99):.4f}  "
            f"temperature(T={temperature:.3f}) LogLoss={temp_valid_nll:.4f}  "
            f"→ using {normalization_winner}"
        )

        X_test = prepare_anonymous_X(test_df)
        raw_test = model.predict_proba(X_test)[:, 1]
        if normalization_winner == "temperature":
            raw_score_test = model.predict(X_test, raw_score=True)
            norm_test = normalize_raw_scores_with_temperature(test_df, raw_score_test, temperature)
        else:
            norm_test = normalize_predictions(test_df, raw_test, method=normalization_winner)
        lgbm_metrics = compute_group_metrics(test_df, norm_test)
        all_metrics["Anonymous LightGBM"] = lgbm_metrics
        print(
            f"\n  Anonymous LightGBM test: Top1={lgbm_metrics.get('top1_acc', 0):.3f} "
            f"Top3={lgbm_metrics.get('top3_acc', 0):.3f} "
            f"LogLoss={lgbm_metrics.get('log_loss', 99):.4f}"
        )

        subgroup_metrics = _anonymous_subgroup_metrics(test_df, norm_test)
        if not args.no_shap:
            print("\n  Computing SHAP on up to 5,000 test rows...")
            shap_mean, feature_names = _compute_anonymous_shap(model, test_df)
    else:
        print("\n  Skipping LightGBM (--no-lgbm)")

    content = generate_anonymous_report(
        seasons=seasons,
        user_profile_id=args.user_profile_id,
        val_stats=val_stats,
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
