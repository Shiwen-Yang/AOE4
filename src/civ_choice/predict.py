"""predict_civ_distribution: runtime inference for one player."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from aoe4_predict.config import DB_PATH, NEW_PLAYER_THRESHOLD
from aoe4_predict.db import get_conn

# Fallback: 18 concrete civs + random-civ option
_N_CIVS = 19
RANDOM_CIV = "random_civ"


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
        SELECT
            CASE
                WHEN civilization_randomized = TRUE THEN '{RANDOM_CIV}'
                ELSE civilization
            END AS civ,
            MIN(started_at) AS first_seen_at
        FROM participants p JOIN games g ON p.game_id = g.game_id
        WHERE civilization IS NOT NULL
        GROUP BY 1
        HAVING MIN(started_at) <= TIMESTAMP '{as_of_str}'
    ),
    player_hist AS (
        SELECT
               p.game_id,
               CASE
                   WHEN p.civilization_randomized = TRUE THEN '{RANDOM_CIV}'
                   ELSE p.civilization
               END AS civ,
               p.result::INT AS result,
               g.started_at, g.patch, g.map
        FROM participants p JOIN games g ON p.game_id = g.game_id
        WHERE p.profile_id = {profile_id}
          AND g.kind IN ('rm_1v1', 'rm_solo')
          AND p.result IS NOT NULL
          AND p.civilization IS NOT NULL
          AND g.started_at < TIMESTAMP '{as_of_str}'
    ),
    prev_civ AS (
        SELECT civ AS last_civ FROM player_hist ORDER BY started_at DESC, game_id DESC LIMIT 1
    ),
    ordered_hist AS (
        SELECT
            civ,
            map,
            ROW_NUMBER() OVER (ORDER BY started_at DESC, game_id DESC) AS rn
        FROM player_hist
    ),
    prev_civ_streak AS (
        SELECT
            oh.civ,
            COUNT(*) AS streak_len
        FROM ordered_hist oh
        WHERE NOT EXISTS (
            SELECT 1
            FROM ordered_hist prior
            WHERE prior.rn < oh.rn
              AND prior.civ <> oh.civ
        )
        GROUP BY oh.civ
    ),
    recent_with_prev AS (
        SELECT
            rn,
            civ,
            map,
            LAG(civ) OVER (ORDER BY rn) AS prev_recent_civ
        FROM ordered_hist
        WHERE rn <= 20
    ),
    counts10 AS (
        SELECT civ, COUNT(*) AS n
        FROM ordered_hist
        WHERE rn <= 10
        GROUP BY civ
    ),
    totals10 AS (
        SELECT SUM(n) AS total FROM counts10
    ),
    entropy10 AS (
        SELECT
            COALESCE(
                -SUM((n::DOUBLE / NULLIF(total, 0)) * LN(n::DOUBLE / NULLIF(total, 0))),
                0
            ) AS entropy
        FROM counts10, totals10
    ),
    last_20_games AS (
        SELECT civ
        FROM player_hist
        ORDER BY started_at DESC, game_id DESC
        LIMIT 20
    )
    SELECT
        vc.civ AS candidate_civ,
        COUNT(ph.civ) AS cand_games_lifetime,
        COALESCE(SUM(ph.result), 0) AS cand_wins_lifetime,
        COUNT(*) FILTER (
            WHERE ph.started_at >= TIMESTAMP '{as_of_str}' - INTERVAL '30 days'
        ) AS cand_games_30d,
        (
            SELECT COUNT(*)
            FROM last_20_games l20
            WHERE l20.civ = vc.civ
        ) AS cand_games_last_20_games,
        (
            SELECT COUNT(*)
            FROM ordered_hist oh
            WHERE oh.rn <= 1 AND oh.civ = vc.civ
        ) AS cand_games_last_1_games,
        (
            SELECT COUNT(*)
            FROM ordered_hist oh
            WHERE oh.rn <= 2 AND oh.civ = vc.civ
        ) AS cand_games_last_2_games,
        (
            SELECT COUNT(*)
            FROM ordered_hist oh
            WHERE oh.rn <= 3 AND oh.civ = vc.civ
        ) AS cand_games_last_3_games,
        (
            SELECT COUNT(*)
            FROM ordered_hist oh
            WHERE oh.rn <= 5 AND oh.civ = vc.civ
        ) AS cand_games_last_5_games,
        (
            SELECT COUNT(*)
            FROM ordered_hist oh
            WHERE oh.rn <= 10 AND oh.civ = vc.civ
        ) AS cand_games_last_10_games,
        COALESCE((
            SELECT MIN(rn)
            FROM ordered_hist oh
            WHERE oh.rn <= 20 AND oh.civ = vc.civ
        ), 21) AS candidate_last_played_position,
        COALESCE((
            SELECT MAX(streak_len)
            FROM prev_civ_streak
        ), 0) AS player_current_streak_len,
        COALESCE((
            SELECT streak_len
            FROM prev_civ_streak pcs
            WHERE pcs.civ = vc.civ
        ), 0) AS candidate_current_streak_len,
        COALESCE((
            SELECT COUNT(*)
            FROM recent_with_prev rwp
            WHERE rwp.rn <= 10
              AND rwp.prev_recent_civ IS NOT NULL
              AND rwp.civ <> rwp.prev_recent_civ
        ), 0) AS recent_civ_switch_count_last_10_games,
        COALESCE((
            SELECT COUNT(DISTINCT civ)
            FROM ordered_hist oh
            WHERE oh.rn <= 10
        ), 0) AS recent_unique_civs_last_10_games,
        COALESCE((SELECT entropy FROM entropy10), 0) AS recent_entropy_last_10_games,
        (
            SELECT COUNT(*)
            FROM recent_with_prev rwp
            WHERE rwp.rn <= 20
              AND rwp.map = '{map_name or ""}'
              AND rwp.civ = vc.civ
        ) AS cand_games_last_20_same_map,
        (
            SELECT COUNT(*)
            FROM recent_with_prev rwp
            WHERE rwp.rn <= 20
              AND rwp.map = '{map_name or ""}'
        ) AS player_games_last_20_same_map,
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
    model_meta: dict[str, Any] | None = None,
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

    if model is None:
        from .model import MODEL_PATH, load_model
        if MODEL_PATH.exists():
            model, model_meta = load_model()

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
    feat_df["cand_global_pr_patch_mmr_bucket"] = 1.0 / _N_CIVS
    feat_df["cand_global_pr_map_patch"] = 1.0 / _N_CIVS

    feat_df = add_derived_features(feat_df)
    X = prepare_X(feat_df)

    from .model import predict_candidate_probs

    normalization = (model_meta or {}).get("normalization", "renorm")
    if normalization == "temperature":
        temperature = float((model_meta or {}).get("temperature") or 1.0)
        if hasattr(model, "predict_proba"):
            raw_scores = model.predict(X, raw_score=True)
        else:
            raw_scores = model.predict(X, raw_score=True)
        scaled = np.asarray(raw_scores, dtype=float) / temperature
        exp_scores = np.exp(np.clip(scaled - scaled.max(), -50, 50))
        norm_probs = exp_scores / exp_scores.sum() if exp_scores.sum() > 0 else np.ones(len(exp_scores)) / len(exp_scores)
    else:
        raw_probs = predict_candidate_probs(model, X)
        if normalization == "softmax":
            raw_probs = np.exp(np.clip(raw_probs, -20, 20))
        total = raw_probs.sum()
        norm_probs = raw_probs / total if total > 0 else np.ones(len(raw_probs)) / len(raw_probs)

    return {
        civ: round(float(p), 4)
        for civ, p in sorted(
            zip(feat_df["candidate_civ"], norm_probs),
            key=lambda x: -x[1],
        )
    }
