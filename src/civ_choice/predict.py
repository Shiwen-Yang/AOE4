"""predict_civ_distribution: runtime inference for one player."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from aoe4_predict.config import DB_PATH, NEW_PLAYER_THRESHOLD
from aoe4_predict.db import get_conn

# Fallback: uniform prior
_N_CIVS = 18


def _query_player_civ_features(
    conn,
    profile_id: int,
    map_name: str | None,
    patch: str | None,
    as_of: datetime,
) -> pd.DataFrame:
    """Return one row per candidate civ with raw stats for this player."""
    as_of_str = as_of.strftime("%Y-%m-%d %H:%M:%S")

    sql = f"""
    WITH
    valid_civs AS (
        SELECT civilization AS civ, MIN(started_at) AS first_seen_at
        FROM participants p JOIN games g ON p.game_id = g.game_id
        WHERE civilization IS NOT NULL
        GROUP BY civilization
        HAVING MIN(started_at) <= TIMESTAMP '{as_of_str}'
    ),
    player_hist AS (
        SELECT p.civilization AS civ, p.result::INT AS result,
               g.started_at, g.patch, g.map
        FROM participants p JOIN games g ON p.game_id = g.game_id
        WHERE p.profile_id = {profile_id}
          AND g.kind IN ('rm_1v1', 'rm_solo')
          AND p.result IS NOT NULL
          AND p.civilization IS NOT NULL
          AND p.civilization_randomized = FALSE
          AND g.started_at < TIMESTAMP '{as_of_str}'
    ),
    prev_civ AS (
        SELECT civ AS last_civ FROM player_hist ORDER BY started_at DESC LIMIT 1
    )
    SELECT
        vc.civ AS candidate_civ,
        COUNT(ph.civ) AS cand_games_lifetime,
        COALESCE(SUM(ph.result), 0) AS cand_wins_lifetime,
        COUNT(*) FILTER (
            WHERE ph.started_at >= TIMESTAMP '{as_of_str}' - INTERVAL '30 days'
        ) AS cand_games_30d,
        COUNT(*) FILTER (WHERE ph.patch = '{patch or ""}') AS cand_games_this_patch,
        COUNT(*) FILTER (WHERE ph.map = '{map_name or ""}') AS cand_games_this_map,
        COALESCE(MAX(DATEDIFF('day', ph.started_at, TIMESTAMP '{as_of_str}')), NULL)
            AS days_since_cand_civ,
        (COUNT(ph.civ) > 0)::INT AS candidate_is_in_pool_lifetime,
        -- Is this civ the player's most recent game civ?
        ((SELECT last_civ FROM prev_civ) = vc.civ)::INT AS candidate_is_last_civ
    FROM valid_civs vc
    LEFT JOIN player_hist ph ON ph.civ = vc.civ
    GROUP BY vc.civ
    """
    return conn.execute(sql).df()


def predict_civ_distribution(
    profile_id: int,
    map_name: str | None = None,
    patch: str | None = None,
    as_of: datetime | None = None,
    conn=None,
    db_path: str | None = None,
    model=None,
) -> dict[str, float]:
    """Return {civ: probability} normalized over all valid candidate civs.

    Falls back to uniform distribution for new players (< NEW_PLAYER_THRESHOLD games).
    """
    if as_of is None:
        as_of = datetime.utcnow()

    own_conn = conn is None
    if own_conn:
        conn = get_conn(db_path or DB_PATH, read_only=True)

    try:
        feat_df = _query_player_civ_features(conn, profile_id, map_name, patch, as_of)
    finally:
        if own_conn:
            conn.close()

    total_games = int(feat_df["cand_games_lifetime"].sum())

    if total_games < NEW_PLAYER_THRESHOLD or model is None:
        # Fallback: uniform distribution
        n = len(feat_df)
        return {row["candidate_civ"]: round(1.0 / n, 4) for _, row in feat_df.iterrows()}

    # Enrich with derived features for model scoring
    feat_df["player_games_lifetime"] = total_games
    feat_df["player_games_30d"] = feat_df["cand_games_30d"].sum()
    feat_df["player_games_this_patch"] = feat_df["cand_games_this_patch"].sum()
    feat_df["player_games_this_map"] = feat_df["cand_games_this_map"].sum()
    feat_df["map"] = map_name or "unknown"
    feat_df["patch"] = patch or "unknown"
    feat_df["season"] = "unknown"
    feat_df["player_mmr"] = np.nan
    feat_df["player_rating"] = np.nan
    feat_df["prev_civ"] = feat_df.loc[feat_df["candidate_is_last_civ"] == 1, "candidate_civ"].iloc[0] \
        if (feat_df["candidate_is_last_civ"] == 1).any() else None

    from .features import add_derived_features, prepare_X
    feat_df["game_id"] = 0
    feat_df["profile_id"] = profile_id
    feat_df["target"] = 0
    feat_df["chosen_civ"] = ""
    feat_df["result"] = 0
    feat_df["started_at"] = as_of
    feat_df["cand_wins_30d"] = 0
    feat_df["cand_wins_this_patch"] = 0
    feat_df["cand_wins_this_map"] = 0
    feat_df["cand_global_pr_prev_season"] = 1.0 / _N_CIVS
    feat_df["cand_global_pr_prev_patch"] = 1.0 / _N_CIVS
    feat_df["cand_global_pr_this_season"] = 1.0 / _N_CIVS

    feat_df = add_derived_features(feat_df)
    X = prepare_X(feat_df)

    raw_probs = model.predict_proba(X)[:, 1]
    total = raw_probs.sum()
    norm_probs = raw_probs / total if total > 0 else np.ones(len(raw_probs)) / len(raw_probs)

    return {
        civ: round(float(p), 4)
        for civ, p in sorted(
            zip(feat_df["candidate_civ"], norm_probs),
            key=lambda x: -x[1],
        )
    }
