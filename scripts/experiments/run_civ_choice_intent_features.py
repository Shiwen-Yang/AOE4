"""Evaluate v2 player-intent civ-choice features against the intent baseline.

The current baseline is the 43-feature intent model saved at
reports/generated/civ_choice_intent_features_lgbm.json.  This runner keeps the
same bounded sample and temporal split, compares that feature set to the v2
recent-sequence feature set, tunes LightGBM on validation log loss, and writes a
compact JSON comparison.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from civ_choice.features import ALL_FEATURES, CONTEXT_FEATURES, add_derived_features
from civ_choice.model import compute_group_metrics, normalize_predictions, temporal_split


ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = ROOT / "reports/generated/civ_choice_tuned_lgbm_xgb_renorm_comparison.json"
INTENT_BASELINE_PATH = ROOT / "reports/generated/civ_choice_intent_features_lgbm.json"
OUT_PATH = ROOT / "reports/generated/civ_choice_intent_v2_lgbm.json"
N_CIV_OPTIONS = 19
RANDOM_CIV = "random_civ"

BASELINE_EXCLUDED_FEATURES = {
    "candidate_was_played_last_3_games",
    "candidate_was_played_last_5_games",
}
INTENT_V1_FEATURES = {
    "cand_pick_share_last_20_games",
    "cand_recent_vs_lifetime_pick_share_delta",
    "candidate_current_streak_len",
}
INTENT_V2_FEATURES = {
    "candidate_played_last_1_game",
    "candidate_played_last_2_games",
    "candidate_was_played_last_3_games",
    "candidate_was_played_last_5_games",
    "candidate_games_last_5_games",
    "candidate_games_last_10_games",
    "candidate_last_played_position",
    "candidate_breaks_current_streak",
    "cand_games_last_20_same_map",
    "cand_pick_share_last_20_same_map",
    "cand_global_pr_patch_mmr_bucket",
    "cand_global_pr_map_patch",
    "player_current_streak_len",
    "recent_civ_switch_count_last_10_games",
    "recent_unique_civs_last_10_games",
    "recent_entropy_last_10_games",
}

INTENT_V1_TUNED_PARAMS = {
    "num_leaves": 75,
    "min_child_samples": 182,
    "learning_rate": 0.026378226388043956,
    "feature_fraction": 0.7056937753654446,
    "bagging_fraction": 0.8277243706586127,
    "bagging_freq": 1,
    "lambda_l1": 3.0377242595071916,
    "lambda_l2": 1.3641929894983322,
    "n_estimators": 700,
}

V2_RAW_COLUMNS = {
    "cand_games_last_1_games",
    "cand_games_last_2_games",
    "cand_games_last_3_games",
    "cand_games_last_5_games",
    "cand_games_last_10_games",
    "candidate_last_played_position",
    "player_current_streak_len",
    "recent_civ_switch_count_last_10_games",
    "recent_unique_civs_last_10_games",
    "recent_entropy_last_10_games",
    "cand_games_last_20_same_map",
    "player_games_last_20_same_map",
    "cand_global_pr_patch_mmr_bucket",
    "cand_global_pr_map_patch",
}


def _has_column(conn: duckdb.DuckDBPyConnection, table: str, column: str) -> bool:
    cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    return column in cols


def _entropy_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return float(
        -sum((n / total) * math.log(n / total) for n in counts.values() if n > 0)
    )


def _backfill_v2_from_history(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    where: str,
) -> pd.DataFrame:
    """Backfill v2 raw sequence columns for older matrices without rebuilding."""
    print("Backfilling v2 sequence features from player history...", flush=True)
    hist_sql = f"""
    WITH matrix AS (
        SELECT DISTINCT profile_id
        FROM civ_choice_training_matrix
        WHERE {where}
    )
    SELECT
        p.game_id,
        p.profile_id,
        CASE
            WHEN p.civilization_randomized = TRUE THEN '{RANDOM_CIV}'
            ELSE p.civilization
        END AS civ,
        g.started_at,
        g.map
    FROM participants p
    JOIN games g ON p.game_id = g.game_id
    JOIN matrix m ON m.profile_id = p.profile_id
    WHERE g.kind IN ('rm_1v1', 'rm_solo')
      AND p.result IS NOT NULL
      AND p.civilization IS NOT NULL
      AND g.started_at IS NOT NULL
    ORDER BY p.profile_id, g.started_at, p.game_id
    """
    hist = conn.execute(hist_sql).df()
    print(f"Loaded {len(hist):,} player-history rows for v2 backfill", flush=True)

    df = df.reset_index(drop=True)
    group_cols = ["game_id", "profile_id"]
    row_groups = df.groupby(group_cols, sort=False, observed=True).indices
    group_meta = (
        df[group_cols + ["started_at", "map"]]
        .drop_duplicates(group_cols)
        .sort_values(["profile_id", "started_at", "game_id"])
    )

    n = len(df)
    arrays = {
        "cand_games_last_1_games": np.zeros(n, dtype=np.int16),
        "cand_games_last_2_games": np.zeros(n, dtype=np.int16),
        "cand_games_last_3_games": np.zeros(n, dtype=np.int16),
        "cand_games_last_5_games": np.zeros(n, dtype=np.int16),
        "cand_games_last_10_games": np.zeros(n, dtype=np.int16),
        "candidate_last_played_position": np.full(n, 21, dtype=np.int16),
        "player_current_streak_len": np.zeros(n, dtype=np.int16),
        "candidate_current_streak_len": np.zeros(n, dtype=np.int16),
        "recent_civ_switch_count_last_10_games": np.zeros(n, dtype=np.int16),
        "recent_unique_civs_last_10_games": np.zeros(n, dtype=np.int16),
        "recent_entropy_last_10_games": np.zeros(n, dtype=np.float32),
        "cand_games_last_20_same_map": np.zeros(n, dtype=np.int16),
        "player_games_last_20_same_map": np.zeros(n, dtype=np.int16),
        "cand_global_pr_patch_mmr_bucket": np.full(n, 1.0 / N_CIV_OPTIONS, dtype=np.float32),
        "cand_global_pr_map_patch": np.full(n, 1.0 / N_CIV_OPTIONS, dtype=np.float32),
    }

    hist_by_profile = {
        pid: sub[["started_at", "game_id", "civ", "map"]].to_records(index=False)
        for pid, sub in hist.groupby("profile_id", sort=False)
    }
    candidate_values = df["candidate_civ"].astype(object).astype(str).values

    processed = 0
    for pid, profile_groups in group_meta.groupby("profile_id", sort=False):
        history = hist_by_profile.get(pid)
        pointer = 0
        if history is None:
            history = []
        for row in profile_groups.itertuples(index=False):
            gid = row.game_id
            started_at = row.started_at
            map_name = row.map
            while pointer < len(history) and history[pointer].started_at < started_at:
                pointer += 1

            recent20 = list(reversed(history[max(0, pointer - 20):pointer]))
            recent10 = recent20[:10]
            civs20 = [str(r.civ) for r in recent20]
            maps20 = [str(r.map) for r in recent20]
            civs10 = civs20[:10]

            counts1 = Counter(civs20[:1])
            counts2 = Counter(civs20[:2])
            counts3 = Counter(civs20[:3])
            counts5 = Counter(civs20[:5])
            counts10 = Counter(civs10)
            same_map_mask = [m == str(map_name) for m in maps20]
            player_same_map = sum(same_map_mask)
            same_map_counts = Counter(
                civ for civ, is_same_map in zip(civs20, same_map_mask) if is_same_map
            )
            last_pos = {}
            for i, civ in enumerate(civs20, start=1):
                last_pos.setdefault(civ, i)

            streak_len = 0
            if civs20:
                last_civ = civs20[0]
                for civ in civs20:
                    if civ != last_civ:
                        break
                    streak_len += 1
            switch_count = sum(
                1 for i in range(1, len(civs10)) if civs10[i] != civs10[i - 1]
            )
            unique10 = len(counts10)
            entropy10 = _entropy_from_counts(counts10)

            idxs = row_groups[(gid, pid)]
            arrays["player_current_streak_len"][idxs] = streak_len
            arrays["recent_civ_switch_count_last_10_games"][idxs] = switch_count
            arrays["recent_unique_civs_last_10_games"][idxs] = unique10
            arrays["recent_entropy_last_10_games"][idxs] = entropy10
            arrays["player_games_last_20_same_map"][idxs] = player_same_map
            for idx in idxs:
                civ = candidate_values[idx]
                arrays["cand_games_last_1_games"][idx] = counts1.get(civ, 0)
                arrays["cand_games_last_2_games"][idx] = counts2.get(civ, 0)
                arrays["cand_games_last_3_games"][idx] = counts3.get(civ, 0)
                arrays["cand_games_last_5_games"][idx] = counts5.get(civ, 0)
                arrays["cand_games_last_10_games"][idx] = counts10.get(civ, 0)
                arrays["candidate_last_played_position"][idx] = last_pos.get(civ, 21)
                arrays["cand_games_last_20_same_map"][idx] = same_map_counts.get(civ, 0)
                if civs20 and civ == civs20[0]:
                    arrays["candidate_current_streak_len"][idx] = streak_len

            processed += 1
            if processed % 50000 == 0:
                print(f"  Backfilled {processed:,} player-matches...", flush=True)

    for col, values in arrays.items():
        df[col] = values
    print(f"Backfilled v2 features for {processed:,} player-matches", flush=True)
    return df


def _load_matrix(conn: duckdb.DuckDBPyConnection, hash_mod: int, hash_remainder: int) -> pd.DataFrame:
    where = f"hash(game_id) % {hash_mod} = {hash_remainder}"
    matrix_cols = [row[1] for row in conn.execute("PRAGMA table_info(civ_choice_training_matrix)").fetchall()]
    has_v2 = V2_RAW_COLUMNS.issubset(matrix_cols)
    if has_v2:
        sql = f"SELECT * FROM civ_choice_training_matrix WHERE {where}"
    else:
        sql = f"SELECT * FROM civ_choice_training_matrix WHERE {where}"
    print(f"Loading civ_choice_training_matrix where {where}...", flush=True)
    df = conn.execute(sql).df()
    if not has_v2:
        df = _backfill_v2_from_history(conn, df, where)
    for col in ("candidate_civ", "chosen_civ", "prev_civ", "map", "patch"):
        if col in df.columns:
            df[col] = df[col].astype("category")
    print(
        f"Loaded {len(df):,} candidate rows across "
        f"{df.groupby(['game_id', 'profile_id']).ngroups:,} player-matches "
        f"(v2_raw_materialized={has_v2})",
        flush=True,
    )
    return df


def _prepare_X(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for col in CONTEXT_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype(object).fillna("missing").astype(str).astype("category")
    return X


def _train_eval(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    params: dict,
    label: str,
) -> tuple[dict, lgb.LGBMClassifier]:
    p = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbose": -1,
        "random_state": 42,
        **params,
    }
    n_estimators = int(p.pop("n_estimators"))
    cat_cols = [c for c in CONTEXT_FEATURES if c in feature_cols]

    X_train = _prepare_X(train_df, feature_cols)
    X_valid = _prepare_X(valid_df, feature_cols)
    X_test = _prepare_X(test_df, feature_cols)
    y_train = train_df["target"].values
    y_valid = valid_df["target"].values

    t0 = time.time()
    model = lgb.LGBMClassifier(n_estimators=n_estimators, **p)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        categorical_feature=cat_cols,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    print(
        f"{label}: best_iter={model.best_iteration_} elapsed={time.time() - t0:.1f}s",
        flush=True,
    )

    valid_probs = normalize_predictions(valid_df, model.predict_proba(X_valid)[:, 1], method="renorm")
    test_probs = normalize_predictions(test_df, model.predict_proba(X_test)[:, 1], method="renorm")
    metrics = {
        "best_iteration": int(model.best_iteration_ or n_estimators),
        "valid": compute_group_metrics(valid_df, valid_probs),
        "test": compute_group_metrics(test_df, test_probs),
    }
    return metrics, model


def _tune_lgbm(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    n_trials: int,
) -> dict:
    X_train = _prepare_X(train_df, feature_cols)
    X_valid = _prepare_X(valid_df, feature_cols)
    y_train = train_df["target"].values
    y_valid = valid_df["target"].values
    cat_cols = [c for c in CONTEXT_FEATURES if c in feature_cols]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbose": -1,
            "random_state": 42,
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 40, 220),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.9),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.65, 0.95),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 8.0),
        }
        t0 = time.time()
        model = lgb.LGBMClassifier(n_estimators=700, **params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            categorical_feature=cat_cols,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        valid_probs = normalize_predictions(
            valid_df,
            model.predict_proba(X_valid)[:, 1],
            method="renorm",
        )
        valid_logloss = compute_group_metrics(valid_df, valid_probs)["log_loss"]
        trial.set_user_attr("best_iteration", int(model.best_iteration_ or 700))
        print(
            f"intent_trial_{trial.number}: logloss={valid_logloss:.6f} "
            f"best_iter={model.best_iteration_} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
        return valid_logloss

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials)
    best = dict(study.best_params)
    best["n_estimators"] = 700
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(ROOT / "aoe4.duckdb"))
    parser.add_argument("--hash-mod", type=int, default=20)
    parser.add_argument("--hash-remainder", type=int, default=0)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--output", default=str(OUT_PATH))
    args = parser.parse_args()

    t0 = time.time()
    conn = duckdb.connect(args.db, read_only=True)
    try:
        df_raw = _load_matrix(conn, args.hash_mod, args.hash_remainder)
    finally:
        conn.close()

    print("Adding derived features...", flush=True)
    df = add_derived_features(df_raw)
    train_df, valid_df, test_df = temporal_split(df)

    baseline_features = [
        f
        for f in ALL_FEATURES
        if f not in BASELINE_EXCLUDED_FEATURES and f not in INTENT_V2_FEATURES
    ]
    intent_features = list(ALL_FEATURES)

    print(f"Training intent-v1 baseline feature set ({len(baseline_features)} features)...", flush=True)
    baseline_metrics, _ = _train_eval(
        train_df,
        valid_df,
        test_df,
        baseline_features,
        INTENT_V1_TUNED_PARAMS,
        "intent_v1_lgbm_fixed_tuned_params",
    )

    print(f"Tuning intent-v2 feature set ({len(intent_features)} features, {args.trials} trials)...", flush=True)
    best_params = _tune_lgbm(train_df, valid_df, intent_features, args.trials)
    print(f"Best intent-v2 params: {best_params}", flush=True)
    intent_metrics, _ = _train_eval(
        train_df,
        valid_df,
        test_df,
        intent_features,
        best_params,
        "intent_v2_lgbm_tuned",
    )

    saved_baseline = None
    if INTENT_BASELINE_PATH.exists():
        saved_baseline = json.loads(INTENT_BASELINE_PATH.read_text()).get("intent_lgbm")

    result = {
        "feature_change": {
            "new_features": sorted(INTENT_V2_FEATURES),
            "baseline_excluded_features": sorted(BASELINE_EXCLUDED_FEATURES),
            "baseline_feature_count": len(baseline_features),
            "intent_v2_feature_count": len(intent_features),
        },
        "normalization": "renorm",
        "dataset": {
            "sample": {
                "hash_mod": args.hash_mod,
                "hash_remainder": args.hash_remainder,
            },
            "train_groups": int(train_df.groupby(["game_id", "profile_id"]).ngroups),
            "valid_groups": int(valid_df.groupby(["game_id", "profile_id"]).ngroups),
            "test_groups": int(test_df.groupby(["game_id", "profile_id"]).ngroups),
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "test_rows": int(len(test_df)),
        },
        "saved_intent_v1_baseline": saved_baseline,
        "rerun_intent_v1_baseline": {
            "params": INTENT_V1_TUNED_PARAMS,
            **baseline_metrics,
        },
        "intent_v2_lgbm": {
            "n_trials": args.trials,
            "best_params": best_params,
            **intent_metrics,
        },
        "deltas_vs_rerun_intent_v1_baseline": {
            "valid_log_loss": (
                intent_metrics["valid"]["log_loss"]
                - baseline_metrics["valid"]["log_loss"]
            ),
            "test_log_loss": (
                intent_metrics["test"]["log_loss"]
                - baseline_metrics["test"]["log_loss"]
            ),
            "test_top1_acc": (
                intent_metrics["test"]["top1_acc"]
                - baseline_metrics["test"]["top1_acc"]
            ),
            "test_top3_acc": (
                intent_metrics["test"]["top3_acc"]
                - baseline_metrics["test"]["top3_acc"]
            ),
            "test_brier": (
                intent_metrics["test"]["brier"]
                - baseline_metrics["test"]["brier"]
            ),
        },
        "elapsed_seconds": time.time() - t0,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out}", flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
