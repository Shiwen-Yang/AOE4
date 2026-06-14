"""Ablate intent-v2 civ-choice feature families.

This reuses the standard bounded sample and temporal split.  Each ablation uses
the same tuned LightGBM params as the v2 model, so the result measures feature
family dependence rather than retuning capacity.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import sys

import duckdb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from civ_choice.features import ALL_FEATURES, add_derived_features
from civ_choice.model import temporal_split

from scripts.experiments.run_civ_choice_intent_features import (
    INTENT_V1_TUNED_PARAMS,
    OUT_PATH as V2_RESULT_PATH,
    _load_matrix,
    _train_eval,
)


OUT_PATH = ROOT / "reports/generated/civ_choice_intent_v2_ablation.json"

FEATURE_FAMILIES = {
    "exact_last_n": [
        "candidate_played_last_1_game",
        "candidate_played_last_2_games",
        "candidate_was_played_last_3_games",
        "candidate_was_played_last_5_games",
        "candidate_games_last_5_games",
        "candidate_games_last_10_games",
        "candidate_last_played_position",
    ],
    "streak": [
        "player_current_streak_len",
        "candidate_current_streak_len",
        "candidate_breaks_current_streak",
        "candidate_is_last_civ",
    ],
    "switch_entropy": [
        "recent_civ_switch_count_last_10_games",
        "recent_unique_civs_last_10_games",
        "recent_entropy_last_10_games",
    ],
    "same_map_recent": [
        "cand_games_last_20_same_map",
        "cand_pick_share_last_20_same_map",
    ],
    "global_priors": [
        "cand_global_pr_patch_mmr_bucket",
        "cand_global_pr_map_patch",
        "cand_global_pr_prev_season",
        "cand_global_pr_prev_patch",
    ],
}


def _load_v2_params() -> dict:
    if V2_RESULT_PATH.exists():
        result = json.loads(V2_RESULT_PATH.read_text())
        params = result.get("intent_v2_lgbm", {}).get("best_params")
        if params:
            return params
    return INTENT_V1_TUNED_PARAMS


def main() -> None:
    t0 = time.time()
    conn = duckdb.connect(str(ROOT / "aoe4.duckdb"), read_only=True)
    try:
        df_raw = _load_matrix(conn, hash_mod=20, hash_remainder=0)
    finally:
        conn.close()

    print("Adding derived features...", flush=True)
    df = add_derived_features(df_raw)
    train_df, valid_df, test_df = temporal_split(df)

    params = _load_v2_params()
    full_features = list(ALL_FEATURES)
    full_metrics, _ = _train_eval(
        train_df,
        valid_df,
        test_df,
        full_features,
        params,
        "intent_v2_full_for_ablation",
    )

    ablations = {}
    for family, drop_features in FEATURE_FAMILIES.items():
        feature_cols = [f for f in full_features if f not in set(drop_features)]
        print(
            f"Ablating {family}: drop {len(drop_features)} features, "
            f"train with {len(feature_cols)} features",
            flush=True,
        )
        metrics, _ = _train_eval(
            train_df,
            valid_df,
            test_df,
            feature_cols,
            params,
            f"ablate_{family}",
        )
        ablations[family] = {
            "dropped_features": drop_features,
            "feature_count": len(feature_cols),
            **metrics,
            "delta_vs_full": {
                "valid_log_loss": metrics["valid"]["log_loss"] - full_metrics["valid"]["log_loss"],
                "test_log_loss": metrics["test"]["log_loss"] - full_metrics["test"]["log_loss"],
                "test_top1_acc": metrics["test"]["top1_acc"] - full_metrics["test"]["top1_acc"],
                "test_top3_acc": metrics["test"]["top3_acc"] - full_metrics["test"]["top3_acc"],
                "test_brier": metrics["test"]["brier"] - full_metrics["test"]["brier"],
            },
        }

    result = {
        "normalization": "renorm",
        "params": params,
        "dataset": {
            "sample": {"hash_mod": 20, "hash_remainder": 0},
            "train_groups": int(train_df.groupby(["game_id", "profile_id"]).ngroups),
            "valid_groups": int(valid_df.groupby(["game_id", "profile_id"]).ngroups),
            "test_groups": int(test_df.groupby(["game_id", "profile_id"]).ngroups),
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "test_rows": int(len(test_df)),
        },
        "full": {
            "feature_count": len(full_features),
            **full_metrics,
        },
        "ablations": ablations,
        "elapsed_seconds": time.time() - t0,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"Wrote {OUT_PATH}", flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
