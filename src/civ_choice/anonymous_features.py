"""Feature preparation for anonymous opponent civ-choice prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd

N_CIV_OPTIONS = 19
SMOOTHING = 10.0

ANONYMOUS_CONTEXT_FEATURES = [
    "candidate_civ",
    "map",
    "patch",
    "season",
    "mmr_bucket",
    "rating_bucket",
]

ANONYMOUS_NUMERIC_FEATURES = [
    "player_mmr",
    "player_rating",
    "cand_global_pr_prior",
    "cand_global_pr_mmr_bucket",
    "cand_global_pr_rating_bucket",
    "cand_global_pr_map",
    "cand_global_pr_map_patch",
    "cand_global_pr_patch",
    "cand_user_recent_opp_pr_10",
    "cand_user_recent_opp_pr_30",
    "cand_user_recent_opp_pr_50",
    "cand_user_recent_opp_pr_same_map_30",
    "user_recent_opp_games_10",
    "user_recent_opp_games_30",
    "user_recent_opp_games_50",
    "user_recent_opp_same_map_games_30",
]

ANONYMOUS_FEATURES = ANONYMOUS_NUMERIC_FEATURES + ANONYMOUS_CONTEXT_FEATURES

FORBIDDEN_PLAYER_HISTORY_FEATURES = {
    "prev_civ",
    "cand_games_lifetime",
    "cand_wins_lifetime",
    "cand_pick_share_lifetime",
    "cand_games_30d",
    "cand_pick_share_30d",
    "candidate_is_last_civ",
    "candidate_last_played_position",
    "player_games_lifetime",
    "player_games_30d",
    "civ_pool_entropy_30d",
    "main_civ_share_lifetime",
}


def _smooth_share(count: pd.Series, total: pd.Series, prior: pd.Series | float) -> pd.Series:
    return (count + SMOOTHING * prior) / (total + SMOOTHING)


def add_anonymous_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add model-ready anonymous features.

    The input matrix should already contain leakage-safe aggregate counts.
    This function intentionally does not derive any target-player history
    features.
    """
    df = df.copy()
    if FORBIDDEN_PLAYER_HISTORY_FEATURES.intersection(df.columns):
        bad = sorted(FORBIDDEN_PLAYER_HISTORY_FEATURES.intersection(df.columns))
        raise ValueError(f"anonymous matrix contains player-history columns: {bad}")

    defaults = {
        "player_mmr": np.nan,
        "player_rating": np.nan,
        "cand_global_prior_count": 0,
        "global_prior_total": 0,
        "cand_mmr_bucket_count": 0,
        "mmr_bucket_total": 0,
        "cand_rating_bucket_count": 0,
        "rating_bucket_total": 0,
        "cand_map_count": 0,
        "map_total": 0,
        "cand_map_patch_count": 0,
        "map_patch_total": 0,
        "cand_patch_count": 0,
        "patch_total": 0,
        "cand_user_recent_opp_count_10": 0,
        "user_recent_opp_games_10": 0,
        "cand_user_recent_opp_count_30": 0,
        "user_recent_opp_games_30": 0,
        "cand_user_recent_opp_count_50": 0,
        "user_recent_opp_games_50": 0,
        "cand_user_recent_opp_same_map_count_30": 0,
        "user_recent_opp_same_map_games_30": 0,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    uniform = 1.0 / N_CIV_OPTIONS
    prior = np.where(
        df["global_prior_total"] > 0,
        df["cand_global_prior_count"] / df["global_prior_total"],
        uniform,
    )
    df["cand_global_pr_prior"] = prior
    df["cand_global_pr_mmr_bucket"] = _smooth_share(
        df["cand_mmr_bucket_count"], df["mmr_bucket_total"], df["cand_global_pr_prior"]
    )
    df["cand_global_pr_rating_bucket"] = _smooth_share(
        df["cand_rating_bucket_count"], df["rating_bucket_total"], df["cand_global_pr_prior"]
    )
    df["cand_global_pr_map"] = _smooth_share(
        df["cand_map_count"], df["map_total"], df["cand_global_pr_prior"]
    )
    df["cand_global_pr_map_patch"] = _smooth_share(
        df["cand_map_patch_count"], df["map_patch_total"], df["cand_global_pr_map"]
    )
    df["cand_global_pr_patch"] = _smooth_share(
        df["cand_patch_count"], df["patch_total"], df["cand_global_pr_prior"]
    )

    mmr_prior = df["cand_global_pr_mmr_bucket"]
    for window in (10, 30, 50):
        df[f"cand_user_recent_opp_pr_{window}"] = _smooth_share(
            df[f"cand_user_recent_opp_count_{window}"],
            df[f"user_recent_opp_games_{window}"],
            mmr_prior,
        )
    df["cand_user_recent_opp_pr_same_map_30"] = _smooth_share(
        df["cand_user_recent_opp_same_map_count_30"],
        df["user_recent_opp_same_map_games_30"],
        df["cand_global_pr_map"],
    )

    for col in ANONYMOUS_FEATURES:
        if col not in df.columns:
            df[col] = 0 if col in ANONYMOUS_NUMERIC_FEATURES else "unknown"

    return df


def prepare_anonymous_X(df: pd.DataFrame) -> pd.DataFrame:
    """Return anonymous feature matrix ready for LightGBM."""
    X = df[ANONYMOUS_FEATURES].copy()
    for col in ANONYMOUS_NUMERIC_FEATURES:
        X[col] = X[col].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for col in ANONYMOUS_CONTEXT_FEATURES:
        X[col] = X[col].astype(object).fillna("unknown").astype(str).astype("category")
    return X
