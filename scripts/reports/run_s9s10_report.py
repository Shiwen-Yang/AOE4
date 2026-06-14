"""
Full pipeline: S9+S10 train/valid, S11 test holdout.
Steps:
  1. Build features (S9, S10, S11)
  2. Tune LightGBM with Optuna (50 trials)
  3. Tune XGBoost with Optuna (50 trials)
  4. Generate analysis report (Enhanced Logistic, Lasso, LightGBM, XGBoost)
"""
import gc
import time
from pathlib import Path

TRAIN_SEASONS = [9, 10]
TEST_SEASONS  = [11]
N_TRIALS      = 50

LGBM_MODEL = Path("models/aoe4_predict/lgbm_s9s10_test_s11.txt")
LGBM_META  = Path("models/aoe4_predict/lgbm_s9s10_test_s11_meta.json")
XGB_MODEL  = Path("models/aoe4_predict/xgb_s9s10_test_s11.ubj")
XGB_META   = Path("models/aoe4_predict/xgb_s9s10_test_s11_meta.json")
REPORT     = Path("reports/generated/analysis_report_s9s10_test_s11.md")


def main():
    from aoe4_predict.db import get_conn
    from aoe4_predict.features import (
        build_civ_matchup_priors, build_player_stats, build_training_features,
    )
    from aoe4_predict.features_extra import (
        extend_training_features, FAMILY_FEATURES, DISABLED_FAMILIES,
    )
    from aoe4_predict.tune import run_tune
    from aoe4_predict.report import generate_report

    all_seasons = sorted(set(TRAIN_SEASONS) | set(TEST_SEASONS))
    families = set(FAMILY_FEATURES.keys()) - DISABLED_FAMILIES

    t_total = time.time()

    # ── 1. Build features ──────────────────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Building features for seasons", all_seasons)
    print("=" * 60)
    conn = get_conn(None)

    print("\n1a. Building player_stats...")
    build_player_stats(conn)

    print("\n1b. Building civ matchup priors...")
    build_civ_matchup_priors(conn)

    print("\n1c. Building training features...")
    df = build_training_features(conn, train_seasons=all_seasons)

    print(f"\n1d. Adding extended feature families: {sorted(families)}")
    del df
    gc.collect()
    conn.close()
    conn = get_conn(None)
    df = extend_training_features(conn, None, families)
    conn.close()
    print(f"  Dataset: {len(df):,} rows × {len(df.columns)} cols")

    # ── 2. Tune LightGBM ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Step 2: Tuning LightGBM ({N_TRIALS} trials, test=S{TEST_SEASONS})")
    print("=" * 60)
    lgbm_params = run_tune(
        df,
        model_type="lgbm",
        n_trials=N_TRIALS,
        retrain=True,
        test_seasons=TEST_SEASONS,
        model_path=LGBM_MODEL,
        meta_path=LGBM_META,
    )

    # ── 3. Tune XGBoost ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Step 3: Tuning XGBoost ({N_TRIALS} trials, test=S{TEST_SEASONS})")
    print("=" * 60)
    xgb_params = run_tune(
        df,
        model_type="xgb",
        n_trials=N_TRIALS,
        retrain=True,
        test_seasons=TEST_SEASONS,
        model_path=XGB_MODEL,
        meta_path=XGB_META,
    )

    # ── 4. Generate report ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4: Generating analysis report")
    print("=" * 60)
    generate_report(
        report_path=REPORT,
        model_path=LGBM_MODEL,
        meta_path=LGBM_META,
    )

    print(f"\n{'=' * 60}")
    print(f"All done in {(time.time() - t_total) / 60:.1f} min")
    print(f"Report: {REPORT}")
    print(f"LightGBM: {LGBM_MODEL}")
    print(f"XGBoost:  {XGB_MODEL}")


if __name__ == "__main__":
    main()
