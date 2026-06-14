"""
Derive model-ready features from the raw training matrix.

All operations are pandas groupby/vectorized — no extra DB access.
"""
import numpy as np
import pandas as pd

N_CIV_OPTIONS = 19  # 18 concrete civs + random_civ option
SMOOTHING = 5  # additive smoothing for win rates and pick shares

# ── Feature name lists ────────────────────────────────────────────────────────

CANDIDATE_FEATURES = [
    "cand_pick_share_lifetime",
    "cand_games_30d",
    "cand_pick_share_30d",
    "cand_games_this_patch",
    "cand_pick_share_this_patch",
    "cand_games_this_map",
    "cand_pick_share_this_map",
    "cand_wr_lifetime",
    "cand_wr_30d",
    "cand_wr_this_patch",
    "cand_wr_this_map",
    "days_since_cand_civ",
    "candidate_is_last_civ",
    "candidate_was_played_last_3_games",
    "candidate_was_played_last_5_games",
    "candidate_played_last_1_game",
    "candidate_played_last_2_games",
    "candidate_games_last_5_games",
    "candidate_games_last_10_games",
    "candidate_last_played_position",
    "candidate_is_most_picked_last_20_games",
    "candidate_is_2nd_most_picked_last_20_games",
    "candidate_is_3rd_most_picked_last_20_games",
    "cand_pick_share_last_20_games",
    "cand_recent_vs_lifetime_pick_share_delta",
    "candidate_current_streak_len",
    "candidate_breaks_current_streak",
    "cand_games_last_20_same_map",
    "cand_pick_share_last_20_same_map",
    "cand_global_pr_patch_mmr_bucket",
    "cand_global_pr_map_patch",
    "candidate_is_lifetime_main",
    "candidate_is_recent_30d_main",
    "candidate_is_patch_main",
    "candidate_civ_rank_lifetime",
    "candidate_civ_rank_30d",
    "candidate_civ_rank_this_patch",
    "candidate_is_in_pool_lifetime",
    "candidate_is_in_pool_30d",
    "cand_global_pr_prev_season",
    "cand_global_pr_prev_patch",
]

PLAYER_FEATURES = [
    "player_rating",
    "player_games_lifetime",
    "player_games_30d",
    "player_games_this_patch",
    "player_games_this_map",
    "num_civs_played_lifetime",
    "num_civs_played_30d",
    "num_civs_played_this_patch",
    "civ_pool_entropy_30d",
    "player_current_streak_len",
    "recent_civ_switch_count_last_10_games",
    "recent_unique_civs_last_10_games",
    "recent_entropy_last_10_games",
    "main_civ_share_lifetime",
]

CONTEXT_FEATURES = ["candidate_civ", "map", "patch", "season"]

ALL_FEATURES = CANDIDATE_FEATURES + PLAYER_FEATURES + CONTEXT_FEATURES


def _entropy(shares: np.ndarray) -> float:
    """Shannon entropy over a probability vector (ignores zeros)."""
    p = shares[shares > 0]
    return float(-np.sum(p * np.log(p + 1e-12)))


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all model-ready features to the training matrix in-place."""
    df = df.copy()
    defaults = {
        "cand_games_last_1_games": 0,
        "cand_games_last_2_games": 0,
        "cand_games_last_3_games": 0,
        "cand_games_last_5_games": 0,
        "cand_games_last_10_games": 0,
        "cand_games_last_20_games": 0,
        "candidate_current_streak_len": 0,
        "player_current_streak_len": 0,
        "candidate_last_played_position": 21,
        "recent_civ_switch_count_last_10_games": 0,
        "recent_unique_civs_last_10_games": 0,
        "recent_entropy_last_10_games": 0.0,
        "cand_games_last_20_same_map": 0,
        "player_games_last_20_same_map": 0,
        "cand_global_pr_patch_mmr_bucket": 1.0 / N_CIV_OPTIONS,
        "cand_global_pr_map_patch": 1.0 / N_CIV_OPTIONS,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    grp = df.groupby(["game_id", "profile_id"])

    # ── Pick shares (smoothed) ────────────────────────────────────────────
    def _smooth_share(games, total, smooth=SMOOTHING):
        return (games + smooth * (1.0 / N_CIV_OPTIONS)) / (total + smooth)

    df["cand_pick_share_lifetime"] = _smooth_share(
        df["cand_games_lifetime"], df["player_games_lifetime"]
    )
    df["cand_pick_share_30d"] = _smooth_share(
        df["cand_games_30d"], df["player_games_30d"]
    )
    df["cand_pick_share_this_patch"] = _smooth_share(
        df["cand_games_this_patch"], df["player_games_this_patch"]
    )
    df["cand_pick_share_this_map"] = _smooth_share(
        df["cand_games_this_map"], df["player_games_this_map"]
    )
    last20_total = grp["cand_games_last_20_games"].transform("sum")
    df["cand_pick_share_last_20_games"] = np.where(
        last20_total > 0,
        df["cand_games_last_20_games"] / last20_total,
        0.0,
    )
    df["cand_recent_vs_lifetime_pick_share_delta"] = (
        df["cand_pick_share_last_20_games"] - df["cand_pick_share_lifetime"]
    )
    df["cand_pick_share_last_20_same_map"] = np.where(
        df["player_games_last_20_same_map"] > 0,
        df["cand_games_last_20_same_map"] / df["player_games_last_20_same_map"],
        0.0,
    )

    # ── Win rates (smoothed) ──────────────────────────────────────────────
    def _smooth_wr(wins, games, smooth=SMOOTHING):
        return (wins + smooth * 0.5) / (games + smooth)

    df["cand_wr_lifetime"] = _smooth_wr(df["cand_wins_lifetime"], df["cand_games_lifetime"])
    df["cand_wr_30d"] = _smooth_wr(df["cand_wins_30d"], df["cand_games_30d"])
    df["cand_wr_this_patch"] = _smooth_wr(df["cand_wins_this_patch"], df["cand_games_this_patch"])
    df["cand_wr_this_map"] = _smooth_wr(df["cand_wins_this_map"], df["cand_games_this_map"])

    # ── Boolean flags ─────────────────────────────────────────────────────
    candidate_civ_values = df["candidate_civ"].astype(object).astype(str)
    prev_civ_values = df["prev_civ"].astype(object).fillna("").astype(str)
    df["candidate_is_last_civ"] = (candidate_civ_values == prev_civ_values).astype(int)
    df["candidate_is_in_pool_lifetime"] = (df["cand_games_lifetime"] > 0).astype(int)
    df["candidate_is_in_pool_30d"] = (df["cand_games_30d"] > 0).astype(int)

    # ── Exact recent sequence flags ───────────────────────────────────────
    df["candidate_played_last_1_game"] = (df["cand_games_last_1_games"] > 0).astype(int)
    df["candidate_played_last_2_games"] = (df["cand_games_last_2_games"] > 0).astype(int)
    df["candidate_was_played_last_3_games"] = (df["cand_games_last_3_games"] > 0).astype(int)
    df["candidate_was_played_last_5_games"] = (df["cand_games_last_5_games"] > 0).astype(int)
    df["candidate_games_last_5_games"] = df["cand_games_last_5_games"]
    df["candidate_games_last_10_games"] = df["cand_games_last_10_games"]
    df["candidate_breaks_current_streak"] = (
        (df["player_current_streak_len"] > 0)
        & (candidate_civ_values != prev_civ_values)
    ).astype(int)

    # ── Within-group ranks and main-civ flags ─────────────────────────────
    df["candidate_civ_rank_lifetime"] = (
        grp["cand_games_lifetime"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    df["candidate_civ_rank_30d"] = (
        grp["cand_games_30d"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    df["candidate_civ_rank_this_patch"] = (
        grp["cand_games_this_patch"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    df["candidate_civ_rank_last_20_games"] = (
        grp["cand_games_last_20_games"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    df["candidate_is_lifetime_main"] = (df["candidate_civ_rank_lifetime"] == 1).astype(int)
    df["candidate_is_recent_30d_main"] = (df["candidate_civ_rank_30d"] == 1).astype(int)
    df["candidate_is_patch_main"] = (df["candidate_civ_rank_this_patch"] == 1).astype(int)
    df["candidate_is_most_picked_last_20_games"] = (
        (df["cand_games_last_20_games"] > 0)
        & (df["candidate_civ_rank_last_20_games"] == 1)
    ).astype(int)
    df["candidate_is_2nd_most_picked_last_20_games"] = (
        (df["cand_games_last_20_games"] > 0)
        & (df["candidate_civ_rank_last_20_games"] == 2)
    ).astype(int)
    df["candidate_is_3rd_most_picked_last_20_games"] = (
        (df["cand_games_last_20_games"] > 0)
        & (df["candidate_civ_rank_last_20_games"] == 3)
    ).astype(int)

    # ── Player-level entropy and pool stats ───────────────────────────────
    def _group_entropy_and_pool(sub):
        n = len(sub)
        shares_lt = sub["cand_pick_share_lifetime"].values
        shares_30 = sub["cand_pick_share_30d"].values
        entr_lt = _entropy(shares_lt / shares_lt.sum() if shares_lt.sum() > 0 else shares_lt)
        entr_30 = _entropy(shares_30 / shares_30.sum() if shares_30.sum() > 0 else shares_30)
        n_civs_lt = int((sub["cand_games_lifetime"] > 0).sum())
        n_civs_30 = int((sub["cand_games_30d"] > 0).sum())
        n_civs_patch = int((sub["cand_games_this_patch"] > 0).sum())
        main_share_lt = sub["cand_pick_share_lifetime"].max()
        main_share_30 = sub["cand_pick_share_30d"].max()
        return pd.Series({
            "civ_pool_entropy_lifetime": entr_lt,
            "civ_pool_entropy_30d": entr_30,
            "num_civs_played_lifetime": n_civs_lt,
            "num_civs_played_30d": n_civs_30,
            "num_civs_played_this_patch": n_civs_patch,
            "main_civ_share_lifetime": main_share_lt,
            "main_civ_share_30d": main_share_30,
        })

    print("  Computing per-group entropy and pool stats...")
    group_stats = grp.apply(_group_entropy_and_pool, include_groups=False).reset_index()
    df = df.merge(group_stats, on=["game_id", "profile_id"], how="left")

    # Ensure all feature columns exist (fill NaN for edge cases)
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = 0 if col not in CONTEXT_FEATURES else "unknown"

    return df


def prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    """Return feature matrix ready for LightGBM (categoricals as category dtype)."""
    X = df[ALL_FEATURES].copy()
    for col in CONTEXT_FEATURES:
        X[col] = X[col].astype(object).fillna("missing").astype(str).astype("category")
    return X
