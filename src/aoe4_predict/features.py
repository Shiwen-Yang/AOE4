"""
Feature engineering.

Two modes:
  1. build_training_features(conn, train_seasons) → materialises training_features table
  2. get_inference_features(player_a, player_b, ..., conn) → dict for a single prediction

Leakage rules:
  - All player historical stats use ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
  - Civ matchup priors use aggregates from seasons BEFORE the target game's season
  - MMR/rating come directly from the game record (they are pre-game values in the dump)
"""
import time
from typing import Any

import pandas as pd

from .config import GLOBAL_WR_PRIOR, NEW_PLAYER_THRESHOLD, PRIOR_STRENGTH
from .db import get_conn, table_exists

# ── SQL for player temporal stats ─────────────────────────────────────────────

_PLAYER_STATS_SQL = """
CREATE OR REPLACE TABLE player_stats AS
WITH player_game AS (
    SELECT
        p.game_id,
        p.profile_id,
        p.result::INT         AS result,
        p.civilization        AS civ,
        p.civilization_randomized,
        p.rating,
        p.mmr,
        g.started_at,
        g.map,
        g.patch,
        g.season
    FROM participants p
    JOIN games g ON p.game_id = g.game_id
    WHERE g.kind IN ('rm_1v1', 'rm_solo')
      AND p.result IS NOT NULL
      AND g.started_at IS NOT NULL
)
SELECT
    game_id,
    profile_id,
    result,
    civ,
    civilization_randomized,
    rating,
    mmr,
    started_at,
    map,
    patch,
    season,

    -- Cumulative games played BEFORE this match (lifetime)
    (ROW_NUMBER() OVER (
        PARTITION BY profile_id
        ORDER BY started_at, game_id
    ) - 1)                                                      AS games_lifetime_before,

    -- Cumulative wins BEFORE this match (lifetime)
    COALESCE(SUM(result) OVER (
        PARTITION BY profile_id
        ORDER BY started_at, game_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0)                                                       AS wins_lifetime_before,

    -- Games this season before 00:00:00 on the calendar day of this match.
    -- ORDER BY DATE + INTERVAL '1 day' PRECEDING gives a hard midnight cutoff:
    -- frame = rows where date <= match_date - 1 day.  No same-day future info.
    COALESCE(COUNT(*) OVER (
        PARTITION BY profile_id, season
        ORDER BY started_at::DATE
        RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL '1 day' PRECEDING
    ), 0)                                                       AS games_season_before,

    -- Wins this season before 00:00:00 on the calendar day of this match.
    COALESCE(SUM(result) OVER (
        PARTITION BY profile_id, season
        ORDER BY started_at::DATE
        RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL '1 day' PRECEDING
    ), 0)                                                       AS wins_season_before,

    -- Days since previous game (NULL if first game)
    DATEDIFF('day',
        LAG(started_at) OVER (
            PARTITION BY profile_id
            ORDER BY started_at, game_id
        ),
        started_at
    )                                                           AS days_since_last_game,

    -- Civ-specific cumulative games BEFORE this match
    (ROW_NUMBER() OVER (
        PARTITION BY profile_id, civ
        ORDER BY started_at, game_id
    ) - 1)                                                      AS civ_games_before,

    -- Civ-specific cumulative wins BEFORE this match
    COALESCE(SUM(result) OVER (
        PARTITION BY profile_id, civ
        ORDER BY started_at, game_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0)                                                       AS civ_wins_before,

    -- Map-specific cumulative games BEFORE this match
    (ROW_NUMBER() OVER (
        PARTITION BY profile_id, map
        ORDER BY started_at, game_id
    ) - 1)                                                      AS map_games_before,

    -- Map-specific cumulative wins BEFORE this match
    COALESCE(SUM(result) OVER (
        PARTITION BY profile_id, map
        ORDER BY started_at, game_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0)                                                       AS map_wins_before
FROM player_game
"""

_CIV_MATCHUP_PRIORS_SQL = """
CREATE OR REPLACE TABLE civ_matchup_priors AS
WITH pairs AS (
    SELECT
        p1.civ           AS civ_a,
        p2.civ           AS civ_b,
        p1.season,
        p1.result        AS result_a
    FROM player_stats p1
    JOIN player_stats p2
        ON p1.game_id = p2.game_id
       AND p1.profile_id < p2.profile_id
),
by_season AS (
    SELECT civ_a, civ_b, season,
           COUNT(*)       AS games,
           SUM(result_a)  AS wins
    FROM pairs
    GROUP BY civ_a, civ_b, season
)
SELECT
    civ_a, civ_b, season,
    -- Cumulative games and wins from ALL seasons BEFORE this one
    SUM(games) OVER (
        PARTITION BY civ_a, civ_b
        ORDER BY season
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )                                                           AS prior_games,
    SUM(wins) OVER (
        PARTITION BY civ_a, civ_b
        ORDER BY season
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )                                                           AS prior_wins
FROM by_season
"""

_TRAINING_FEATURES_SQL = """
CREATE OR REPLACE TABLE training_features AS
WITH pairs AS (
    SELECT
        a.game_id,
        a.started_at,
        a.map,
        a.patch,
        a.season,
        -- target: 1 = player A wins
        a.result                            AS target,
        -- Player A (lower profile_id)
        a.profile_id                        AS profile_id_a,
        a.civ                               AS civ_a,
        a.civilization_randomized           AS civ_rand_a,
        a.mmr                               AS mmr_a,
        a.rating                            AS rating_a,
        a.games_lifetime_before             AS games_lifetime_a,
        a.wins_lifetime_before              AS wins_lifetime_a,
        a.games_season_before               AS games_season_a,
        a.wins_season_before                AS wins_season_a,
        a.days_since_last_game              AS days_since_a,
        a.civ_games_before                  AS civ_games_a,
        a.civ_wins_before                   AS civ_wins_a,
        a.map_games_before                  AS map_games_a,
        a.map_wins_before                   AS map_wins_a,
        -- Player B (higher profile_id)
        b.profile_id                        AS profile_id_b,
        b.civ                               AS civ_b,
        b.civilization_randomized           AS civ_rand_b,
        b.mmr                               AS mmr_b,
        b.rating                            AS rating_b,
        b.games_lifetime_before             AS games_lifetime_b,
        b.wins_lifetime_before              AS wins_lifetime_b,
        b.games_season_before               AS games_season_b,
        b.wins_season_before                AS wins_season_b,
        b.days_since_last_game              AS days_since_b,
        b.civ_games_before                  AS civ_games_b,
        b.civ_wins_before                   AS civ_wins_b,
        b.map_games_before                  AS map_games_b,
        b.map_wins_before                   AS map_wins_b
    FROM player_stats a
    JOIN player_stats b
        ON a.game_id = b.game_id
       AND a.profile_id < b.profile_id
    WHERE a.season IN ({seasons})
)
SELECT
    p.*,
    mp.prior_games  AS prior_matchup_games,
    mp.prior_wins   AS prior_matchup_wins
FROM pairs p
LEFT JOIN civ_matchup_priors mp
    ON p.civ_a = mp.civ_a
   AND p.civ_b = mp.civ_b
   AND p.season = mp.season
"""


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived and smoothed features in Python from raw SQL counts."""
    p = PRIOR_STRENGTH
    g = GLOBAL_WR_PRIOR

    def smooth(wins, games):
        return (wins + p * g) / (games + p)

    # Smoothed win rates (additive prior, falls back to global_wr when games=0)
    df["overall_wr_a"] = smooth(df["wins_lifetime_a"], df["games_lifetime_a"])
    df["overall_wr_b"] = smooth(df["wins_lifetime_b"], df["games_lifetime_b"])
    df["season_wr_a"] = smooth(df["wins_season_a"], df["games_season_a"])
    df["season_wr_b"] = smooth(df["wins_season_b"], df["games_season_b"])
    df["civ_wr_a"] = smooth(df["civ_wins_a"], df["civ_games_a"])
    df["civ_wr_b"] = smooth(df["civ_wins_b"], df["civ_games_b"])
    df["map_wr_a"] = smooth(df["map_wins_a"], df["map_games_a"])
    df["map_wr_b"] = smooth(df["map_wins_b"], df["map_games_b"])

    # Smoothed civ matchup prior
    df["prior_matchup_wr_a"] = smooth(
        df["prior_matchup_wins"].fillna(0),
        df["prior_matchup_games"].fillna(0),
    )

    # Primary skill signal: MMR preferred, rating as fallback
    df["skill_a"] = df["mmr_a"].where(df["mmr_a"].notna(), df["rating_a"])
    df["skill_b"] = df["mmr_b"].where(df["mmr_b"].notna(), df["rating_b"])

    # Missingness indicators
    df["missing_mmr_a"] = df["mmr_a"].isna().astype(int)
    df["missing_mmr_b"] = df["mmr_b"].isna().astype(int)
    df["missing_rating_a"] = df["rating_a"].isna().astype(int)
    df["missing_rating_b"] = df["rating_b"].isna().astype(int)
    df["missing_skill_a"] = df["skill_a"].isna().astype(int)
    df["missing_skill_b"] = df["skill_b"].isna().astype(int)

    # Difference features
    df["mmr_diff"] = df["mmr_a"] - df["mmr_b"]
    df["rating_diff"] = df["rating_a"] - df["rating_b"]
    df["skill_diff"] = df["skill_a"] - df["skill_b"]
    df["games_diff"] = df["games_lifetime_a"] - df["games_lifetime_b"]
    df["wr_diff"] = df["overall_wr_a"] - df["overall_wr_b"]

    # New/low-history player flags
    df["is_new_player_a"] = (df["games_lifetime_a"] < NEW_PLAYER_THRESHOLD).astype(int)
    df["is_new_player_b"] = (df["games_lifetime_b"] < NEW_PLAYER_THRESHOLD).astype(int)

    # Context availability flags
    df["civs_known"] = (df["civ_a"].notna() & df["civ_b"].notna()).astype(int)
    df["map_known"] = df["map"].notna().astype(int)
    df["full_context_known"] = (df["civs_known"] & df["map_known"]).astype(int)

    return df


def build_player_stats(conn) -> None:
    print("  Building player_stats table (window functions over all data)...", flush=True)
    t0 = time.time()
    conn.execute(_PLAYER_STATS_SQL)
    n = conn.execute("SELECT count(*) FROM player_stats").fetchone()[0]
    print(f"  player_stats: {n:,} rows in {time.time()-t0:.1f}s")


def build_civ_matchup_priors(conn) -> None:
    print("  Building civ_matchup_priors...", flush=True)
    conn.execute(_CIV_MATCHUP_PRIORS_SQL)
    n = conn.execute("SELECT count(*) FROM civ_matchup_priors").fetchone()[0]
    print(f"  civ_matchup_priors: {n:,} rows")


def build_training_features(conn, train_seasons: list[int]) -> pd.DataFrame:
    """
    Materialize training_features table and return as DataFrame.

    The table contains one row per RM 1v1 game in train_seasons,
    with player A assigned as the participant with the lower profile_id.
    All features use only data available before the game's start time.
    """
    seasons_str = ", ".join(str(s) for s in train_seasons)
    sql = _TRAINING_FEATURES_SQL.format(seasons=seasons_str)

    print(f"  Building training_features for seasons [{seasons_str}]...", flush=True)
    t0 = time.time()
    conn.execute(sql)
    n = conn.execute("SELECT count(*) FROM training_features").fetchone()[0]
    print(f"  training_features: {n:,} rows in {time.time()-t0:.1f}s")

    df = conn.execute("SELECT * FROM training_features").df()
    df = _add_derived_features(df)
    return df


# ── Inference feature construction ────────────────────────────────────────────

def _get_player_current_stats(profile_id: int, conn) -> dict[str, Any]:
    """
    Compute a player's current stats using all available DB history.
    Uses two queries to avoid mixing aggregates and window functions.
    """
    agg = conn.execute(
        """
        SELECT
            count(*)           AS games_lifetime,
            sum(p.result::INT) AS wins_lifetime,
            max(g.started_at)  AS last_game_at,
            max(g.season)      AS last_season
        FROM participants p
        JOIN games g ON p.game_id = g.game_id
        WHERE p.profile_id = ?
          AND g.kind IN ('rm_1v1','rm_solo')
          AND p.result IS NOT NULL
        """,
        [profile_id],
    ).fetchone()

    if agg is None or agg[0] == 0:
        return {
            "games_lifetime": 0,
            "wins_lifetime": 0,
            "last_mmr": None,
            "last_rating": None,
            "last_game_at": None,
            "last_season": None,
        }

    # Most recent non-null MMR
    mmr_row = conn.execute(
        """
        SELECT p.mmr, p.rating
        FROM participants p
        JOIN games g ON p.game_id = g.game_id
        WHERE p.profile_id = ? AND g.kind IN ('rm_1v1','rm_solo')
          AND (p.mmr IS NOT NULL OR p.rating IS NOT NULL)
        ORDER BY g.started_at DESC
        LIMIT 1
        """,
        [profile_id],
    ).fetchone()

    return {
        "games_lifetime": agg[0],
        "wins_lifetime": agg[1],
        "last_mmr": mmr_row[0] if mmr_row else None,
        "last_rating": mmr_row[1] if mmr_row else None,
        "last_game_at": agg[2],
        "last_season": agg[3],
    }


def _get_player_season_stats(profile_id: int, season: int, conn) -> dict[str, int]:
    """Season games and wins before midnight today (matches training day-level window).

    g.started_at < current_date: SQL implicitly casts a bare DATE to that date at
    00:00:00 when compared to a timestamp, giving the same midnight cutoff as
    INTERVAL '1 day' PRECEDING in the training SQL.
    """
    row = conn.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(p.result::INT), 0)
        FROM participants p
        JOIN games g ON p.game_id = g.game_id
        WHERE p.profile_id = ?
          AND g.season = ?
          AND g.kind IN ('rm_1v1', 'rm_solo')
          AND p.result IS NOT NULL
          AND g.started_at < current_date
        """,
        [profile_id, season],
    ).fetchone()
    return {"games_season": row[0] or 0, "wins_season": row[1] or 0}


def _get_player_civ_stats(profile_id: int, civ: str | None, conn) -> dict[str, Any]:
    if civ is None:
        return {"civ_games": 0, "civ_wins": 0}
    row = conn.execute(
        """
        SELECT count(*), sum(p.result::INT)
        FROM participants p JOIN games g ON p.game_id = g.game_id
        WHERE p.profile_id = ? AND p.civilization = ?
          AND g.kind IN ('rm_1v1','rm_solo') AND p.result IS NOT NULL
        """,
        [profile_id, civ],
    ).fetchone()
    return {"civ_games": row[0] or 0, "civ_wins": row[1] or 0}


def _get_player_map_stats(profile_id: int, map_name: str | None, conn) -> dict[str, Any]:
    if map_name is None:
        return {"map_games": 0, "map_wins": 0}
    row = conn.execute(
        """
        SELECT count(*), sum(p.result::INT)
        FROM participants p JOIN games g ON p.game_id = g.game_id
        WHERE p.profile_id = ? AND g.map = ?
          AND g.kind IN ('rm_1v1','rm_solo') AND p.result IS NOT NULL
        """,
        [profile_id, map_name],
    ).fetchone()
    return {"map_games": row[0] or 0, "map_wins": row[1] or 0}


def _get_civ_matchup_prior(civ_a: str | None, civ_b: str | None, conn) -> dict[str, Any]:
    if civ_a is None or civ_b is None:
        return {"prior_matchup_games": 0, "prior_matchup_wins": 0}

    # Use civ_a < civ_b ordering to match the priors table pairing convention
    if civ_a > civ_b:
        civ_a, civ_b, flip = civ_b, civ_a, True
    else:
        flip = False

    row = conn.execute(
        """
        SELECT sum(prior_games), sum(prior_wins)
        FROM civ_matchup_priors
        WHERE civ_a = ? AND civ_b = ?
        """,
        [civ_a, civ_b],
    ).fetchone()
    games = row[0] or 0
    wins = row[1] or 0
    if flip:
        wins = games - wins
    return {"prior_matchup_games": games, "prior_matchup_wins": wins}


def get_inference_features(
    player_a_id: int,
    player_b_id: int,
    civ_a: str | None = None,
    civ_b: str | None = None,
    map_name: str | None = None,
    season: int | None = None,
    patch: str | None = None,
    conn=None,
    db_path=None,
) -> dict[str, Any]:
    """
    Compute features for a single match prediction.
    Returns a dict that can be passed to the model after feature_cols selection.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_conn(db_path, read_only=True)

    if season is None:
        season = conn.execute("SELECT max(season) FROM games").fetchone()[0]
    if patch is None:
        patch = conn.execute("SELECT patch FROM games ORDER BY started_at DESC LIMIT 1").fetchone()
        patch = patch[0] if patch else None

    a = _get_player_current_stats(player_a_id, conn)
    b = _get_player_current_stats(player_b_id, conn)
    season_a = _get_player_season_stats(player_a_id, season, conn)
    season_b = _get_player_season_stats(player_b_id, season, conn)
    civ_stats_a = _get_player_civ_stats(player_a_id, civ_a, conn)
    civ_stats_b = _get_player_civ_stats(player_b_id, civ_b, conn)
    map_stats_a = _get_player_map_stats(player_a_id, map_name, conn)
    map_stats_b = _get_player_map_stats(player_b_id, map_name, conn)
    matchup = _get_civ_matchup_prior(civ_a, civ_b, conn)

    import datetime
    now = datetime.datetime.utcnow()

    def days_since(ts):
        if ts is None:
            return None
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        if ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
        return (now - ts).days

    p = PRIOR_STRENGTH
    g = GLOBAL_WR_PRIOR

    def smooth(wins, games):
        return (wins + p * g) / (games + p)

    feat: dict[str, Any] = {
        "season": season,
        "patch": patch,
        "map": map_name,
        "civ_a": civ_a,
        "civ_b": civ_b,
        "civ_rand_a": False,
        "civ_rand_b": False,
        # Player A raw counts
        "mmr_a": a["last_mmr"],
        "rating_a": a["last_rating"],
        "games_lifetime_a": a["games_lifetime"],
        "wins_lifetime_a": a["wins_lifetime"],
        "games_season_a": season_a["games_season"],
        "wins_season_a": season_a["wins_season"],
        "days_since_a": days_since(a["last_game_at"]),
        "civ_games_a": civ_stats_a["civ_games"],
        "civ_wins_a": civ_stats_a["civ_wins"],
        "map_games_a": map_stats_a["map_games"],
        "map_wins_a": map_stats_a["map_wins"],
        # Player B raw counts
        "mmr_b": b["last_mmr"],
        "rating_b": b["last_rating"],
        "games_lifetime_b": b["games_lifetime"],
        "wins_lifetime_b": b["wins_lifetime"],
        "games_season_b": season_b["games_season"],
        "wins_season_b": season_b["wins_season"],
        "days_since_b": days_since(b["last_game_at"]),
        "civ_games_b": civ_stats_b["civ_games"],
        "civ_wins_b": civ_stats_b["civ_wins"],
        "map_games_b": map_stats_b["map_games"],
        "map_wins_b": map_stats_b["map_wins"],
        "prior_matchup_games": matchup["prior_matchup_games"],
        "prior_matchup_wins": matchup["prior_matchup_wins"],
    }

    # Derived features (mirror _add_derived_features)
    feat["overall_wr_a"] = smooth(feat["wins_lifetime_a"], feat["games_lifetime_a"])
    feat["overall_wr_b"] = smooth(feat["wins_lifetime_b"], feat["games_lifetime_b"])
    feat["season_wr_a"] = smooth(feat["wins_season_a"], feat["games_season_a"])
    feat["season_wr_b"] = smooth(feat["wins_season_b"], feat["games_season_b"])
    feat["civ_wr_a"] = smooth(feat["civ_wins_a"], feat["civ_games_a"])
    feat["civ_wr_b"] = smooth(feat["civ_wins_b"], feat["civ_games_b"])
    feat["map_wr_a"] = smooth(feat["map_wins_a"], feat["map_games_a"])
    feat["map_wr_b"] = smooth(feat["map_wins_b"], feat["map_games_b"])
    feat["prior_matchup_wr_a"] = smooth(feat["prior_matchup_wins"], feat["prior_matchup_games"])

    skill_a = feat["mmr_a"] if feat["mmr_a"] is not None else feat["rating_a"]
    skill_b = feat["mmr_b"] if feat["mmr_b"] is not None else feat["rating_b"]
    feat["skill_a"] = skill_a
    feat["skill_b"] = skill_b

    feat["missing_mmr_a"] = int(feat["mmr_a"] is None)
    feat["missing_mmr_b"] = int(feat["mmr_b"] is None)
    feat["missing_rating_a"] = int(feat["rating_a"] is None)
    feat["missing_rating_b"] = int(feat["rating_b"] is None)
    feat["missing_skill_a"] = int(skill_a is None)
    feat["missing_skill_b"] = int(skill_b is None)

    feat["mmr_diff"] = (feat["mmr_a"] or 0) - (feat["mmr_b"] or 0)
    feat["rating_diff"] = (feat["rating_a"] or 0) - (feat["rating_b"] or 0)
    feat["skill_diff"] = (skill_a or 0) - (skill_b or 0)
    feat["games_diff"] = feat["games_lifetime_a"] - feat["games_lifetime_b"]
    feat["wr_diff"] = feat["overall_wr_a"] - feat["overall_wr_b"]

    feat["is_new_player_a"] = int(feat["games_lifetime_a"] < NEW_PLAYER_THRESHOLD)
    feat["is_new_player_b"] = int(feat["games_lifetime_b"] < NEW_PLAYER_THRESHOLD)
    feat["civs_known"] = int(civ_a is not None and civ_b is not None)
    feat["map_known"] = int(map_name is not None)
    feat["full_context_known"] = int(feat["civs_known"] and feat["map_known"])

    if own_conn:
        conn.close()

    return feat
