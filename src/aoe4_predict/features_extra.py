"""
Extended feature families for AOE4 RM 1v1 outcome prediction.

Families implemented:
  P1  civ_recency      — time-windowed civ history (7/30/60d), civ fraction, days since civ
  P2  mmr_trend        — MMR change over last N games, volatility (std), slope proxy
  P3  adjusted_form    — recent win rate over last 5/10/20 games (raw, smoothed)
  P4  duration_profile — short/long game split, average duration by context
  P5  head_to_head     — cumulative prior games + wins between the exact pair
  P8  low_history      — granular missingness and cold-start boolean flags
  P9  activity_session — time-windowed total game counts (7/14/30d), inactivity flags

Families NOT yet implemented (stubs):
  P6  map_archetypes   — requires manual map metadata table
  P7  patch_priors     — complex patch-level aggregate priors
  P10 time_server      — requires hour/server/country data
  P11 elo              — rolling Elo/Glicko computation

Leakage guarantee:
  - All SQL windows use ROWS/RANGE ... 1 PRECEDING (excludes current game).
  - RANGE INTERVAL windows exclude any game at the same timestamp.
  - H2H uses ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING.
  - No post-match information is used.
"""
import time
from typing import Any

import numpy as np
import pandas as pd

from .config import GLOBAL_WR_PRIOR, PRIOR_STRENGTH
from .db import table_exists

# ── SQL: extended per-player-game stats ───────────────────────────────────────
#
# One row per (player, game).  Keys: game_id, profile_id.
# Covers P1 (civ recency), P2 (MMR trend), P3 (recent form),
#        P4 (duration profile), P9 (activity).

_PLAYER_STATS_EXT_SQL = """
CREATE OR REPLACE TABLE player_stats_ext AS
WITH player_game AS (
    SELECT
        p.game_id,
        p.profile_id,
        p.result::INT   AS result,
        p.civilization  AS civ,
        p.mmr,
        g.started_at,
        g.duration
    FROM participants p
    JOIN games g ON p.game_id = g.game_id
    WHERE g.kind IN ('rm_1v1', 'rm_solo')
      AND p.result  IS NOT NULL
      AND g.started_at IS NOT NULL
)
SELECT
    game_id,
    profile_id,
    civ,

    -- ── P1: Civ recency (RANGE windows = time-based, excludes current row) ──

    COUNT(*) OVER (
        PARTITION BY profile_id, civ
        ORDER BY started_at
        RANGE BETWEEN INTERVAL '7 days' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
    ) AS civ_games_7d,

    COALESCE(SUM(result) OVER (
        PARTITION BY profile_id, civ
        ORDER BY started_at
        RANGE BETWEEN INTERVAL '7 days' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
    ), 0) AS civ_wins_7d,

    COUNT(*) OVER (
        PARTITION BY profile_id, civ
        ORDER BY started_at
        RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
    ) AS civ_games_30d,

    COALESCE(SUM(result) OVER (
        PARTITION BY profile_id, civ
        ORDER BY started_at
        RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
    ), 0) AS civ_wins_30d,

    COUNT(*) OVER (
        PARTITION BY profile_id, civ
        ORDER BY started_at
        RANGE BETWEEN INTERVAL '60 days' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
    ) AS civ_games_60d,

    COALESCE(SUM(result) OVER (
        PARTITION BY profile_id, civ
        ORDER BY started_at
        RANGE BETWEEN INTERVAL '60 days' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
    ), 0) AS civ_wins_60d,

    DATEDIFF('day',
        LAG(started_at) OVER (
            PARTITION BY profile_id, civ
            ORDER BY started_at, game_id
        ),
        started_at
    ) AS days_since_civ,

    -- ── P1/P9 shared: all-civ activity denominators ──

    COUNT(*) OVER (
        PARTITION BY profile_id
        ORDER BY started_at
        RANGE BETWEEN INTERVAL '7 days' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
    ) AS act_games_7d,

    COUNT(*) OVER (
        PARTITION BY profile_id
        ORDER BY started_at
        RANGE BETWEEN INTERVAL '14 days' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
    ) AS act_games_14d,

    COUNT(*) OVER (
        PARTITION BY profile_id
        ORDER BY started_at
        RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
    ) AS act_games_30d,

    COUNT(*) OVER (
        PARTITION BY profile_id
        ORDER BY started_at
        RANGE BETWEEN INTERVAL '60 days' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
    ) AS act_games_60d,

    -- ── P2: MMR trend (LAG-based, ROWS windows) ──

    LAG(mmr, 3)  OVER (PARTITION BY profile_id ORDER BY started_at, game_id) AS mmr_lag3,
    LAG(mmr, 5)  OVER (PARTITION BY profile_id ORDER BY started_at, game_id) AS mmr_lag5,
    LAG(mmr, 10) OVER (PARTITION BY profile_id ORDER BY started_at, game_id) AS mmr_lag10,
    LAG(mmr, 20) OVER (PARTITION BY profile_id ORDER BY started_at, game_id) AS mmr_lag20,

    STDDEV_SAMP(mmr) OVER (
        PARTITION BY profile_id
        ORDER BY started_at, game_id
        ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
    ) AS mmr_std_10,

    STDDEV_SAMP(mmr) OVER (
        PARTITION BY profile_id
        ORDER BY started_at, game_id
        ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
    ) AS mmr_std_20,

    -- ── P3: Recent form (ROWS-based) ──

    COUNT(*) OVER (
        PARTITION BY profile_id
        ORDER BY started_at, game_id
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    ) AS recent_n_5,

    COALESCE(SUM(result) OVER (
        PARTITION BY profile_id
        ORDER BY started_at, game_id
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    ), 0) AS recent_w_5,

    COUNT(*) OVER (
        PARTITION BY profile_id
        ORDER BY started_at, game_id
        ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
    ) AS recent_n_10,

    COALESCE(SUM(result) OVER (
        PARTITION BY profile_id
        ORDER BY started_at, game_id
        ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
    ), 0) AS recent_w_10,

    COUNT(*) OVER (
        PARTITION BY profile_id
        ORDER BY started_at, game_id
        ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
    ) AS recent_n_20,

    COALESCE(SUM(result) OVER (
        PARTITION BY profile_id
        ORDER BY started_at, game_id
        ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
    ), 0) AS recent_w_20,

    -- ── P4: Duration profile ──

    AVG(duration) OVER (
        PARTITION BY profile_id
        ORDER BY started_at, game_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS avg_dur_life,

    AVG(duration) OVER (
        PARTITION BY profile_id
        ORDER BY started_at, game_id
        ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
    ) AS avg_dur_20,

    COALESCE(SUM(CASE WHEN duration <= 900  THEN 1    ELSE 0 END) OVER (
        PARTITION BY profile_id
        ORDER BY started_at, game_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0) AS short_games,

    COALESCE(SUM(CASE WHEN duration <= 900  THEN result ELSE 0 END) OVER (
        PARTITION BY profile_id
        ORDER BY started_at, game_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0) AS short_wins,

    COALESCE(SUM(CASE WHEN duration > 1800 THEN 1    ELSE 0 END) OVER (
        PARTITION BY profile_id
        ORDER BY started_at, game_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0) AS long_games,

    COALESCE(SUM(CASE WHEN duration > 1800 THEN result ELSE 0 END) OVER (
        PARTITION BY profile_id
        ORDER BY started_at, game_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0) AS long_wins,

    AVG(duration) OVER (
        PARTITION BY profile_id, civ
        ORDER BY started_at, game_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS civ_avg_dur

FROM player_game
"""

# ── SQL: head-to-head prior stats ─────────────────────────────────────────────
#
# One row per (player_pair, game).
# pid_lo < pid_hi by construction (mirrors training_features convention).
# h2h_games_before: number of prior meetings between the pair.
# h2h_wins_lo_before: cumulative wins by the lower-id player.

_H2H_PRIORS_SQL = """
CREATE OR REPLACE TABLE h2h_priors AS
WITH h2h AS (
    SELECT
        a.profile_id AS pid_lo,
        b.profile_id AS pid_hi,
        a.game_id,
        g.started_at,
        a.result::INT AS lo_wins
    FROM participants a
    JOIN participants b
        ON  a.game_id   = b.game_id
        AND a.profile_id < b.profile_id
    JOIN games g ON a.game_id = g.game_id
    WHERE g.kind IN ('rm_1v1', 'rm_solo') AND a.result IS NOT NULL
)
SELECT
    game_id,
    pid_lo,
    pid_hi,
    (ROW_NUMBER() OVER (
        PARTITION BY pid_lo, pid_hi
        ORDER BY started_at, game_id
    ) - 1)                                                      AS h2h_games_before,
    COALESCE(SUM(lo_wins) OVER (
        PARTITION BY pid_lo, pid_hi
        ORDER BY started_at, game_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0)                                                       AS h2h_wins_lo_before
FROM h2h
"""


# ── build functions ───────────────────────────────────────────────────────────

def build_player_stats_ext(conn) -> None:
    print("  Building player_stats_ext (civ recency, MMR trend, recent form, duration, activity)...", flush=True)
    t0 = time.time()
    conn.execute(_PLAYER_STATS_EXT_SQL)
    n = conn.execute("SELECT count(*) FROM player_stats_ext").fetchone()[0]
    print(f"  player_stats_ext: {n:,} rows in {time.time() - t0:.1f}s")


def build_h2h_priors(conn) -> None:
    print("  Building h2h_priors (head-to-head)...", flush=True)
    t0 = time.time()
    conn.execute(_H2H_PRIORS_SQL)
    n = conn.execute("SELECT count(*) FROM h2h_priors").fetchone()[0]
    print(f"  h2h_priors: {n:,} rows in {time.time() - t0:.1f}s")


# ── fetch extra columns from DB (joins done in DuckDB for efficiency) ─────────

_EXT_SELECT_A = """
    ea.civ_games_7d   AS civ_games_7d_a,
    ea.civ_wins_7d    AS civ_wins_7d_a,
    ea.civ_games_30d  AS civ_games_30d_a,
    ea.civ_wins_30d   AS civ_wins_30d_a,
    ea.civ_games_60d  AS civ_games_60d_a,
    ea.civ_wins_60d   AS civ_wins_60d_a,
    ea.days_since_civ AS days_since_civ_a,
    ea.act_games_7d   AS act_games_7d_a,
    ea.act_games_14d  AS act_games_14d_a,
    ea.act_games_30d  AS act_games_30d_a,
    ea.act_games_60d  AS act_games_60d_a,
    ea.mmr_lag3       AS mmr_lag3_a,
    ea.mmr_lag5       AS mmr_lag5_a,
    ea.mmr_lag10      AS mmr_lag10_a,
    ea.mmr_lag20      AS mmr_lag20_a,
    ea.mmr_std_10     AS mmr_std_10_a,
    ea.mmr_std_20     AS mmr_std_20_a,
    ea.recent_n_5     AS recent_n_5_a,
    ea.recent_w_5     AS recent_w_5_a,
    ea.recent_n_10    AS recent_n_10_a,
    ea.recent_w_10    AS recent_w_10_a,
    ea.recent_n_20    AS recent_n_20_a,
    ea.recent_w_20    AS recent_w_20_a,
    ea.avg_dur_life   AS avg_dur_life_a,
    ea.avg_dur_20     AS avg_dur_20_a,
    ea.short_games    AS short_games_a,
    ea.short_wins     AS short_wins_a,
    ea.long_games     AS long_games_a,
    ea.long_wins      AS long_wins_a,
    ea.civ_avg_dur    AS civ_avg_dur_a
"""

_EXT_SELECT_B = """
    eb.civ_games_7d   AS civ_games_7d_b,
    eb.civ_wins_7d    AS civ_wins_7d_b,
    eb.civ_games_30d  AS civ_games_30d_b,
    eb.civ_wins_30d   AS civ_wins_30d_b,
    eb.civ_games_60d  AS civ_games_60d_b,
    eb.civ_wins_60d   AS civ_wins_60d_b,
    eb.days_since_civ AS days_since_civ_b,
    eb.act_games_7d   AS act_games_7d_b,
    eb.act_games_14d  AS act_games_14d_b,
    eb.act_games_30d  AS act_games_30d_b,
    eb.act_games_60d  AS act_games_60d_b,
    eb.mmr_lag3       AS mmr_lag3_b,
    eb.mmr_lag5       AS mmr_lag5_b,
    eb.mmr_lag10      AS mmr_lag10_b,
    eb.mmr_lag20      AS mmr_lag20_b,
    eb.mmr_std_10     AS mmr_std_10_b,
    eb.mmr_std_20     AS mmr_std_20_b,
    eb.recent_n_5     AS recent_n_5_b,
    eb.recent_w_5     AS recent_w_5_b,
    eb.recent_n_10    AS recent_n_10_b,
    eb.recent_w_10    AS recent_w_10_b,
    eb.recent_n_20    AS recent_n_20_b,
    eb.recent_w_20    AS recent_w_20_b,
    eb.avg_dur_life   AS avg_dur_life_b,
    eb.avg_dur_20     AS avg_dur_20_b,
    eb.short_games    AS short_games_b,
    eb.short_wins     AS short_wins_b,
    eb.long_games     AS long_games_b,
    eb.long_wins      AS long_wins_b,
    eb.civ_avg_dur    AS civ_avg_dur_b
"""


def _fetch_player_ext(conn, has_h2h: bool) -> pd.DataFrame:
    """
    Join player_stats_ext (×2) and optionally h2h_priors with training_features.

    Caller must free the original df and call gc.collect() before calling this,
    because this loads a wide result that includes tf.* plus all extension columns.
    Materializes in DuckDB first (spills to disk if needed) then loads once.
    """
    h2h_select = ""
    h2h_join = ""
    if has_h2h:
        h2h_select = ",\n    h.h2h_games_before AS h2h_games,\n    h.h2h_wins_lo_before AS h2h_wins_a"
        h2h_join = "LEFT JOIN h2h_priors h ON tf.game_id = h.game_id"

    # Pre-filter player_stats_ext to training games (shrinks hash table ~2.5×)
    conn.execute("""
        CREATE OR REPLACE TABLE _pse_train AS
        SELECT pse.*
        FROM player_stats_ext pse
        INNER JOIN training_features tf ON pse.game_id = tf.game_id
    """)

    # Materialize the full wide join in DuckDB (spills to disk if needed)
    conn.execute(f"""
        CREATE OR REPLACE TABLE _training_wide AS
        SELECT
            tf.*,
            {_EXT_SELECT_A},
            {_EXT_SELECT_B}
            {h2h_select}
        FROM training_features tf
        LEFT JOIN _pse_train ea
            ON tf.game_id = ea.game_id AND tf.profile_id_a = ea.profile_id
        LEFT JOIN _pse_train eb
            ON tf.game_id = eb.game_id AND tf.profile_id_b = eb.profile_id
        {h2h_join}
    """)
    conn.execute("DROP TABLE IF EXISTS _pse_train")

    result = conn.execute("SELECT * FROM _training_wide").df()
    conn.execute("DROP TABLE IF EXISTS _training_wide")
    return result


# ── Python-level derived features ─────────────────────────────────────────────

def _smooth(wins, games, p: int = PRIOR_STRENGTH, g: float = GLOBAL_WR_PRIOR):
    return (wins + p * g) / (games + p)


def _p1_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Civ recency: smoothed win rates, civ fraction, diffs."""
    for side in ("a", "b"):
        for days in (7, 30, 60):
            g_col = f"civ_games_{days}d_{side}"
            w_col = f"civ_wins_{days}d_{side}"
            if g_col in df.columns:
                df[f"civ_wr_{days}d_{side}"] = _smooth(df[w_col].fillna(0), df[g_col].fillna(0))

        # Fraction of recent games on this civ (30d)
        g30 = df.get(f"civ_games_30d_{side}", 0)
        total30 = df.get(f"act_games_30d_{side}", 0)
        df[f"civ_frac_30d_{side}"] = g30.fillna(0) / (total30.fillna(0).clip(lower=1))

        # Fraction of recent games on this civ (60d)
        g60 = df.get(f"civ_games_60d_{side}", 0)
        total60 = df.get(f"act_games_60d_{side}", 0)
        df[f"civ_frac_60d_{side}"] = g60.fillna(0) / (total60.fillna(0).clip(lower=1))

    # Differences A−B
    for feat in ("civ_wr_30d", "civ_wr_60d", "civ_frac_30d", "civ_frac_60d"):
        if f"{feat}_a" in df.columns and f"{feat}_b" in df.columns:
            df[f"{feat}_diff"] = df[f"{feat}_a"] - df[f"{feat}_b"]

    return df


def _p2_derived(df: pd.DataFrame) -> pd.DataFrame:
    """MMR trend: change over N games, slope proxy, volatility diff, rising flag."""
    for side in ("a", "b"):
        mmr = df.get(f"mmr_{side}")
        if mmr is None:
            continue
        for n in (3, 5, 10, 20):
            lag_col = f"mmr_lag{n}_{side}"
            if lag_col in df.columns:
                df[f"mmr_change_{n}_{side}"] = mmr - df[lag_col]

        # Slope proxy: change_per_game over last 10
        if f"mmr_change_10_{side}" in df.columns:
            df[f"mmr_slope_10_{side}"] = df[f"mmr_change_10_{side}"] / 10
            df[f"mmr_rising_10_{side}"] = (
                df[f"mmr_change_10_{side}"].fillna(0) > 0
            ).astype(int)
        if f"mmr_change_20_{side}" in df.columns:
            df[f"mmr_slope_20_{side}"] = df[f"mmr_change_20_{side}"] / 20

    # Diffs (A−B)
    for feat in ("mmr_change_10", "mmr_change_20", "mmr_slope_10", "mmr_std_10", "mmr_std_20"):
        if f"{feat}_a" in df.columns and f"{feat}_b" in df.columns:
            df[f"{feat}_diff"] = df[f"{feat}_a"] - df[f"{feat}_b"]

    return df


def _p3_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Recent form: smoothed win rates over last N games, diffs."""
    for side in ("a", "b"):
        for n in (5, 10, 20):
            n_col = f"recent_n_{n}_{side}"
            w_col = f"recent_w_{n}_{side}"
            if n_col in df.columns:
                df[f"recent_wr_{n}_{side}"] = _smooth(
                    df[w_col].fillna(0), df[n_col].fillna(0)
                )

    for n in (5, 10, 20):
        fa, fb = f"recent_wr_{n}_a", f"recent_wr_{n}_b"
        if fa in df.columns and fb in df.columns:
            df[f"recent_wr_{n}_diff"] = df[fa] - df[fb]

    return df


def _p4_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Duration profile: smoothed short/long WR, shares, diffs."""
    for side in ("a", "b"):
        games = df.get(f"games_lifetime_{side}")
        if games is None:
            continue

        for bucket, g_col, w_col in [
            ("short", f"short_games_{side}", f"short_wins_{side}"),
            ("long",  f"long_games_{side}",  f"long_wins_{side}"),
        ]:
            if g_col in df.columns:
                df[f"{bucket}_wr_{side}"] = _smooth(
                    df[w_col].fillna(0), df[g_col].fillna(0)
                )
                df[f"{bucket}_share_{side}"] = (
                    df[g_col].fillna(0) / games.clip(lower=1)
                )

    # Diffs
    for feat in ("avg_dur_life", "avg_dur_20", "short_wr", "long_wr",
                 "short_share", "long_share", "civ_avg_dur"):
        fa, fb = f"{feat}_a", f"{feat}_b"
        if fa in df.columns and fb in df.columns:
            df[f"{feat}_diff"] = df[fa].fillna(0) - df[fb].fillna(0)

    return df


def _p5_derived(df: pd.DataFrame) -> pd.DataFrame:
    """H2H: smoothed win rate with strong prior (strength=5, base=0.5)."""
    if "h2h_games" not in df.columns:
        return df
    H2H_PRIOR = 5
    df["h2h_wr_a"] = (df["h2h_wins_a"].fillna(0) + H2H_PRIOR * 0.5) / (
        df["h2h_games"].fillna(0) + H2H_PRIOR
    )
    df["h2h_has_data"] = (df["h2h_games"].fillna(0) >= 3).astype(int)
    return df


def _p8_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Low-history and missing-skill decomposition flags."""
    ma = df.get("missing_mmr_a", pd.Series(0, index=df.index)).fillna(0).astype(bool)
    mb = df.get("missing_mmr_b", pd.Series(0, index=df.index)).fillna(0).astype(bool)

    df["both_mmr_missing"]   = (ma & mb).astype(int)
    df["a_mmr_missing_only"] = (ma & ~mb).astype(int)
    df["b_mmr_missing_only"] = (~ma & mb).astype(int)

    for side in ("a", "b"):
        g = df.get(f"games_lifetime_{side}", pd.Series(0, index=df.index)).fillna(0)
        for thr in (5, 10, 20):
            df[f"{side}_low_lt{thr}"] = (g < thr).astype(int)

    if "a_low_lt20" in df.columns and "b_low_lt20" in df.columns:
        df["one_low_one_est"] = (
            (df["a_low_lt20"] == 1) ^ (df["b_low_lt20"] == 1)
        ).astype(int)
        df["both_low"] = (
            (df["a_low_lt20"] == 1) & (df["b_low_lt20"] == 1)
        ).astype(int)

    for side in ("a", "b"):
        gs = df.get(f"games_season_{side}", pd.Series(0, index=df.index)).fillna(0)
        df[f"is_first10_season_{side}"] = (gs < 10).astype(int)

    return df


def _p9_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Activity: inactivity flags, activity diffs."""
    for side in ("a", "b"):
        g30 = df.get(f"act_games_30d_{side}", pd.Series(0, index=df.index)).fillna(0)
        df[f"inactive_30d_{side}"] = (g30 == 0).astype(int)
        g7 = df.get(f"act_games_7d_{side}", pd.Series(0, index=df.index)).fillna(0)
        df[f"inactive_7d_{side}"] = (g7 == 0).astype(int)

    for window in (7, 14, 30):
        fa = f"act_games_{window}d_a"
        fb = f"act_games_{window}d_b"
        if fa in df.columns and fb in df.columns:
            df[f"act_games_{window}d_diff"] = df[fa].fillna(0) - df[fb].fillna(0)

    return df


# ── SQL: player map-archetype stats (P6) ──────────────────────────────────────
#
# One row per (player, game). Keys: game_id, profile_id.
# Requires map_metadata table to be loaded first (ingest-metadata command).

_PLAYER_MAP_ARCHETYPE_SQL = """
CREATE OR REPLACE TABLE player_map_archetype_stats AS
SELECT
    p.game_id,
    p.profile_id,
    m.map_type_primary,
    m.openness,
    m.chokepoint_level,

    (ROW_NUMBER() OVER (
        PARTITION BY p.profile_id, m.map_type_primary
        ORDER BY g.started_at, p.game_id
    ) - 1) AS map_type_games_before,

    COALESCE(SUM(p.result::INT) OVER (
        PARTITION BY p.profile_id, m.map_type_primary
        ORDER BY g.started_at, p.game_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0) AS map_type_wins_before,

    (ROW_NUMBER() OVER (
        PARTITION BY p.profile_id, m.openness
        ORDER BY g.started_at, p.game_id
    ) - 1) AS openness_games_before,

    COALESCE(SUM(p.result::INT) OVER (
        PARTITION BY p.profile_id, m.openness
        ORDER BY g.started_at, p.game_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0) AS openness_wins_before

FROM participants p
JOIN games g ON p.game_id = g.game_id
LEFT JOIN map_metadata m ON g.map_id = m.map_id
WHERE g.kind IN ('rm_1v1', 'rm_solo')
  AND p.result IS NOT NULL
  AND g.started_at IS NOT NULL
"""

# ── SQL: map-level empirical priors by patch (P7) ─────────────────────────────
#
# One row per (map_id, patch): cumulative stats from ALL patches before this one.
# Ordered by patch_start_at (from patch_metadata) not by patch string.
# Joined to patch_metadata; falls back to MIN(started_at) per patch when CSV absent.

_MAP_PATCH_PRIORS_SQL = """
CREATE OR REPLACE TABLE map_patch_priors AS
WITH patch_dates AS (
    SELECT
        g.patch,
        COALESCE(pm.patch_start_at, MIN(g.started_at)) AS patch_start_at
    FROM games g
    LEFT JOIN patch_metadata pm ON g.patch = pm.patch
    GROUP BY g.patch, pm.patch_start_at
),
by_patch AS (
    SELECT
        g.map_id,
        g.patch,
        pd.patch_start_at,
        COUNT(*)            AS games,
        SUM(g.duration)     AS total_duration,
        SUM(CASE WHEN g.duration <= 900  THEN 1 ELSE 0 END) AS short_games,
        SUM(CASE WHEN g.duration >= 1800 THEN 1 ELSE 0 END) AS long_games
    FROM games g
    JOIN patch_dates pd ON g.patch = pd.patch
    WHERE g.kind = 'rm_1v1' AND g.map_id IS NOT NULL
    GROUP BY g.map_id, g.patch, pd.patch_start_at
)
SELECT
    map_id,
    patch,
    patch_start_at,

    SUM(games) OVER (
        PARTITION BY map_id
        ORDER BY patch_start_at
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS prior_map_games,

    -- True weighted avg (sum of totals / sum of games), not avg-of-avgs
    SUM(total_duration) OVER (
        PARTITION BY map_id
        ORDER BY patch_start_at
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) * 1.0 / NULLIF(
        SUM(games) OVER (
            PARTITION BY map_id
            ORDER BY patch_start_at
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ), 0
    ) AS prior_avg_duration,

    SUM(short_games) OVER (
        PARTITION BY map_id
        ORDER BY patch_start_at
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) * 1.0 / NULLIF(
        SUM(games) OVER (
            PARTITION BY map_id
            ORDER BY patch_start_at
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ), 0
    ) AS prior_short_game_share,

    SUM(long_games) OVER (
        PARTITION BY map_id
        ORDER BY patch_start_at
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) * 1.0 / NULLIF(
        SUM(games) OVER (
            PARTITION BY map_id
            ORDER BY patch_start_at
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ), 0
    ) AS prior_long_game_share

FROM by_patch
"""


def build_player_map_archetype_stats(conn) -> None:
    from .db import table_exists as te
    if not te(conn, "map_metadata"):
        print("  Warning: map_metadata table not found — run 'ingest-metadata' first. Skipping P6 SQL table.")
        return
    print("  Building player_map_archetype_stats (map archetype history)...", flush=True)
    t0 = time.time()
    conn.execute(_PLAYER_MAP_ARCHETYPE_SQL)
    n = conn.execute("SELECT count(*) FROM player_map_archetype_stats").fetchone()[0]
    print(f"  player_map_archetype_stats: {n:,} rows in {time.time() - t0:.1f}s")


def build_map_patch_priors(conn) -> None:
    print("  Building map_patch_priors (empirical map stats by patch)...", flush=True)
    t0 = time.time()
    conn.execute(_MAP_PATCH_PRIORS_SQL)
    n = conn.execute("SELECT count(*) FROM map_patch_priors").fetchone()[0]
    print(f"  map_patch_priors: {n:,} rows in {time.time() - t0:.1f}s")


def _fetch_p6_features(conn) -> pd.DataFrame:
    """Fetch P6 map-archetype player history joined to training_features."""
    query = """
    SELECT
        tf.game_id,
        -- Map metadata (same for both players — from the game)
        m.map_type_primary,
        m.map_geography_subtype,
        m.water_importance,
        m.openness,
        m.chokepoint_level,
        m.trade_viability,
        m.resource_scarcity,
        m.map_metadata_confidence,
        COALESCE(m.map_metadata_missing, CASE WHEN m.map_id IS NULL THEN 1 ELSE 0 END) AS map_metadata_missing,
        -- Player A archetype history
        pa.map_type_games_before AS map_type_games_a,
        pa.map_type_wins_before  AS map_type_wins_a,
        pa.openness_games_before AS openness_games_a,
        pa.openness_wins_before  AS openness_wins_a,
        -- Player B archetype history
        pb.map_type_games_before AS map_type_games_b,
        pb.map_type_wins_before  AS map_type_wins_b,
        pb.openness_games_before AS openness_games_b,
        pb.openness_wins_before  AS openness_wins_b
    FROM training_features tf
    LEFT JOIN map_metadata m ON tf.map = m.map
    LEFT JOIN player_map_archetype_stats pa
        ON tf.game_id = pa.game_id AND tf.profile_id_a = pa.profile_id
    LEFT JOIN player_map_archetype_stats pb
        ON tf.game_id = pb.game_id AND tf.profile_id_b = pb.profile_id
    """
    return conn.execute(query).df()


def _fetch_p7_features(conn) -> pd.DataFrame:
    """Fetch P7 empirical map priors and patch-age features."""
    has_patch_meta = table_exists(conn, "patch_metadata")
    patch_age_sql = ""
    if has_patch_meta:
        patch_age_sql = """
        ,
        DATEDIFF('day', pm.patch_start_at, g.started_at)   AS patch_age_days,
        (DATEDIFF('day', pm.patch_start_at, g.started_at) <= 7)::INT AS is_first_7d_of_patch,
        COALESCE(pm.is_major_patch,   0)::INT AS is_major_patch,
        COALESCE(pm.is_balance_patch, 0)::INT AS is_balance_patch,
        CASE WHEN pm.patch_start_at IS NULL THEN 1 ELSE 0 END AS missing_patch_start
        """
        patch_join = "LEFT JOIN patch_metadata pm ON g.patch = pm.patch"
    else:
        patch_age_sql = ""
        patch_join = ""

    query = f"""
    SELECT
        tf.game_id,
        mpp.prior_map_games,
        LN(COALESCE(mpp.prior_map_games, 0) + 1) AS log_prior_map_games,
        mpp.prior_avg_duration,
        mpp.prior_short_game_share,
        mpp.prior_long_game_share
        {patch_age_sql}
    FROM training_features tf
    JOIN games g ON tf.game_id = g.game_id
    LEFT JOIN map_patch_priors mpp
        ON g.map_id = mpp.map_id AND g.patch = mpp.patch
    {patch_join}
    """
    return conn.execute(query).df()


def _p6_derived(df: pd.DataFrame) -> pd.DataFrame:
    """P6: Smoothed map-type and openness WR for each player, plus diffs."""
    for side in ("a", "b"):
        for dim in ("map_type", "openness"):
            g_col = f"{dim}_games_{side}"
            w_col = f"{dim}_wins_{side}"
            if g_col in df.columns:
                df[f"{dim}_wr_{side}"] = _smooth(df[w_col].fillna(0), df[g_col].fillna(0))

    for feat in ("map_type_wr", "openness_wr"):
        if f"{feat}_a" in df.columns and f"{feat}_b" in df.columns:
            df[f"{feat}_diff"] = df[f"{feat}_a"] - df[f"{feat}_b"]

    return df


def _p7_derived(df: pd.DataFrame) -> pd.DataFrame:
    """P7: Fill nulls for map priors (new maps have no prior data)."""
    # prior_map_games = 0 for first-time patches; log of 0 = 0
    if "prior_map_games" in df.columns:
        df["prior_map_games"] = df["prior_map_games"].fillna(0)
        df["log_prior_map_games"] = np.log1p(df["prior_map_games"])
    return df


# ── main entry point ──────────────────────────────────────────────────────────

#: Families that require player_stats_ext (the ext SQL table)
_EXT_TABLE_FAMILIES = frozenset({"civ_recency", "mmr_trend", "adjusted_form",
                                  "duration_profile", "activity_session"})

#: Unimplemented stubs (reduced — P6/P7 now real)
_STUB_FAMILIES = frozenset({"time_server", "elo"})


def extend_training_features(
    conn,
    df: "pd.DataFrame | None",
    families: set[str],
) -> pd.DataFrame:
    """
    Augment training DataFrame with extended feature families.

    Parameters
    ----------
    conn : duckdb connection (writable — may need to create tables)
    df   : output of build_training_features + _add_derived_features, or None
           to load directly from the 'training_features' DuckDB table (caller
           should have deleted its reference and gc.collect()'d before calling).
    families : set of family names to add, e.g. {'civ_recency', 'mmr_trend'}
               Pass 'all' or an empty set for all/none.

    Returns the augmented DataFrame.
    """
    # _fetch_player_ext reloads from training_features (raw SQL cols, no Python-derived
    # features). _add_derived_features must always run after the join to recreate
    # skill_diff, civ_wr_a, etc. — regardless of whether df was None or not.
    _always_derive = True

    if isinstance(families, str) and families == "all":
        families = _EXT_TABLE_FAMILIES | {"head_to_head", "low_history_detail",
                                           "map_archetypes", "patch_priors"}

    for stub in _STUB_FAMILIES & families:
        print(f"  Warning: {stub} not yet implemented — skipping.")

    active = families - _STUB_FAMILIES

    needs_ext = bool(active & _EXT_TABLE_FAMILIES)
    needs_h2h = "head_to_head" in active
    needs_p6  = "map_archetypes" in active
    needs_p7  = "patch_priors" in active

    # Build materialized tables if needed
    if needs_ext and not table_exists(conn, "player_stats_ext"):
        build_player_stats_ext(conn)
    if needs_h2h and not table_exists(conn, "h2h_priors"):
        build_h2h_priors(conn)
    if needs_p6 and not table_exists(conn, "player_map_archetype_stats"):
        build_player_map_archetype_stats(conn)
    if needs_p7 and not table_exists(conn, "map_patch_priors"):
        build_map_patch_priors(conn)

    # Fetch joined extended columns from DuckDB.
    # When df is None (caller released its reference), skip the intermediate Python load
    # and go directly to the wide join — _fetch_player_ext already loads tf.* from DuckDB.
    # When df was passed in, delete it first to free memory before the join.
    if needs_ext or needs_h2h:
        import gc
        print("  Materializing P1-P5/P9 wide join in DuckDB then reloading...", flush=True)
        t0 = time.time()
        if df is not None:
            del df
        gc.collect()
        df = _fetch_player_ext(conn, has_h2h=needs_h2h)
        # Always recompute base derived features — _fetch_player_ext returns raw tf.* cols
        from .features import _add_derived_features
        df = _add_derived_features(df)
        print(f"  Extended join (P1-P5/P9): {len(df):,} rows in {time.time() - t0:.1f}s")
    else:
        # P6/P7/P8 only (no wide join) — load base df from DB
        from .features import _add_derived_features
        print("  Loading training_features from DB...", flush=True)
        df = conn.execute("SELECT * FROM training_features").df()
        df = _add_derived_features(df)

    if needs_p6 and table_exists(conn, "player_map_archetype_stats"):
        print("  Joining P6 map-archetype features...", flush=True)
        t0 = time.time()
        p6_df = _fetch_p6_features(conn)
        df = df.merge(p6_df, on="game_id", how="left")
        print(f"  P6 join: {len(p6_df):,} rows in {time.time() - t0:.1f}s")

    if needs_p7:
        print("  Joining P7 map-prior and patch-age features...", flush=True)
        t0 = time.time()
        p7_df = _fetch_p7_features(conn)
        df = df.merge(p7_df, on="game_id", how="left")
        print(f"  P7 join: {len(p7_df):,} rows in {time.time() - t0:.1f}s")

    # Python-level derived features (always fast, no DB needed)
    if "civ_recency" in active:
        df = _p1_derived(df)
    if "mmr_trend" in active:
        df = _p2_derived(df)
    if "adjusted_form" in active:
        df = _p3_derived(df)
    if "duration_profile" in active:
        df = _p4_derived(df)
    if "head_to_head" in active:
        df = _p5_derived(df)
    if "low_history_detail" in active:
        df = _p8_derived(df)
    if "activity_session" in active:
        df = _p9_derived(df)
    if "map_archetypes" in active:
        df = _p6_derived(df)
    if "patch_priors" in active:
        df = _p7_derived(df)

    # Drop columns that belong to non-active families.  The wide SQL join
    # (_fetch_player_ext) returns ALL columns from player_stats_ext regardless of
    # which families were requested; stripping unused columns here keeps the returned
    # DataFrame clean and prevents disabled-family features from leaking into training.
    cols_to_drop = []
    for fam_name, feats in FAMILY_FEATURES.items():
        if fam_name not in active:
            for feat in feats:
                if feat in df.columns:
                    cols_to_drop.append(feat)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Downcast float64 → float32 to halve the returned DataFrame footprint.
    # LightGBM bins to uint8 internally, so float32 precision is sufficient.
    for col in df.select_dtypes(include="float64").columns:
        df[col] = df[col].astype("float32")
    import gc; gc.collect()

    return df


# ── inference-time extended features (P1-P5, P8-P9) ──────────────────────────

def get_extended_inference_features(
    player_a_id: int,
    player_b_id: int,
    civ_a: str | None,
    civ_b: str | None,
    base_feat: dict,
    conn,
    before_timestamp=None,
) -> dict[str, Any]:
    """
    Compute P1-P5, P8-P9 extended features for a single match prediction.

    Queries participants+games directly — no dependency on player_stats_ext or
    h2h_priors being pre-built.  Mirrors the window semantics used during training:
      - Time-window counts (P1, P9) use before_timestamp as the reference point
        (falls back to now() for live predictions).
      - Lag-based features (P2, P3) use the N most recent games in chronological
        order, matching LAG(mmr, N) / ROWS BETWEEN N PRECEDING AND 1 PRECEDING.
      - Lifetime duration stats (P4) cover all recorded games before the cutoff.

    Returns a dict of additional features; caller merges with base_feat.
    The player_a / player_b assignment in base_feat must already follow the
    profile_id_a < profile_id_b convention.
    """
    import datetime

    # Reference point for time-window features (P1, P9)
    ref_ts = before_timestamp if before_timestamp is not None else datetime.datetime.utcnow()
    # Ensure ref_ts is a datetime (not a pandas Timestamp)
    if hasattr(ref_ts, "to_pydatetime"):
        ref_ts = ref_ts.to_pydatetime()
    if hasattr(ref_ts, "tzinfo") and ref_ts.tzinfo is not None:
        ref_ts = ref_ts.replace(tzinfo=None)

    ts_upper_clause = "AND g.started_at < $ts" if before_timestamp is not None else ""

    ext: dict[str, Any] = {}
    pid_lo = min(player_a_id, player_b_id)
    pid_hi = max(player_a_id, player_b_id)
    lo_is_a = player_a_id == pid_lo

    for side, pid, civ in (("a", player_a_id, civ_a), ("b", player_b_id, civ_b)):
        # ── One aggregation query covers P1 (civ windows), P4 (lifetime duration
        #    stats), and P9 (activity windows).  civ=None causes all civ FILTER
        #    conditions to evaluate as NULL → 0, which is the correct default.
        agg = conn.execute(f"""
            SELECT
                AVG(g.duration)                                                                                  AS avg_dur_life,
                AVG(g.duration) FILTER (WHERE p.civilization = $civ)                                            AS civ_avg_dur,
                COUNT(*)        FILTER (WHERE g.duration IS NOT NULL AND g.duration <= 900)                      AS short_games,
                COALESCE(SUM(p.result::INT) FILTER (WHERE g.duration IS NOT NULL AND g.duration <= 900),  0)    AS short_wins,
                COUNT(*)        FILTER (WHERE g.duration IS NOT NULL AND g.duration >  1800)                     AS long_games,
                COALESCE(SUM(p.result::INT) FILTER (WHERE g.duration IS NOT NULL AND g.duration >  1800),  0)   AS long_wins,
                COUNT(*)        FILTER (WHERE p.civilization = $civ AND g.started_at >= $ts - INTERVAL '7 days')  AS civ_games_7d,
                COALESCE(SUM(p.result::INT) FILTER (WHERE p.civilization = $civ AND g.started_at >= $ts - INTERVAL '7 days'),  0) AS civ_wins_7d,
                COUNT(*)        FILTER (WHERE p.civilization = $civ AND g.started_at >= $ts - INTERVAL '30 days') AS civ_games_30d,
                COALESCE(SUM(p.result::INT) FILTER (WHERE p.civilization = $civ AND g.started_at >= $ts - INTERVAL '30 days'), 0) AS civ_wins_30d,
                COUNT(*)        FILTER (WHERE p.civilization = $civ AND g.started_at >= $ts - INTERVAL '60 days') AS civ_games_60d,
                COALESCE(SUM(p.result::INT) FILTER (WHERE p.civilization = $civ AND g.started_at >= $ts - INTERVAL '60 days'), 0) AS civ_wins_60d,
                MAX(g.started_at) FILTER (WHERE p.civilization = $civ)                                          AS last_civ_game_at,
                COUNT(*)        FILTER (WHERE g.started_at >= $ts - INTERVAL '7 days')                          AS act_games_7d,
                COUNT(*)        FILTER (WHERE g.started_at >= $ts - INTERVAL '14 days')                         AS act_games_14d,
                COUNT(*)        FILTER (WHERE g.started_at >= $ts - INTERVAL '30 days')                         AS act_games_30d,
                COUNT(*)        FILTER (WHERE g.started_at >= $ts - INTERVAL '60 days')                         AS act_games_60d
            FROM participants p
            JOIN games g ON p.game_id = g.game_id
            WHERE p.profile_id = $pid
              AND g.kind IN ('rm_1v1', 'rm_solo')
              AND p.result IS NOT NULL
              AND g.started_at IS NOT NULL
              {ts_upper_clause}
        """, {"pid": pid, "civ": civ, "ts": ref_ts}).fetchone()

        ext[f"avg_dur_life_{side}"]   = agg[0]
        ext[f"civ_avg_dur_{side}"]    = agg[1]
        ext[f"short_games_{side}"]    = agg[2] or 0
        ext[f"short_wins_{side}"]     = agg[3] or 0
        ext[f"long_games_{side}"]     = agg[4] or 0
        ext[f"long_wins_{side}"]      = agg[5] or 0
        ext[f"civ_games_7d_{side}"]   = agg[6] or 0
        ext[f"civ_wins_7d_{side}"]    = agg[7] or 0
        ext[f"civ_games_30d_{side}"]  = agg[8] or 0
        ext[f"civ_wins_30d_{side}"]   = agg[9] or 0
        ext[f"civ_games_60d_{side}"]  = agg[10] or 0
        ext[f"civ_wins_60d_{side}"]   = agg[11] or 0
        ext[f"act_games_7d_{side}"]   = agg[13] or 0
        ext[f"act_games_14d_{side}"]  = agg[14] or 0
        ext[f"act_games_30d_{side}"]  = agg[15] or 0
        ext[f"act_games_60d_{side}"]  = agg[16] or 0

        last_civ_ts = agg[12]
        if last_civ_ts is not None:
            ts = last_civ_ts
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            if ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            ext[f"days_since_civ_{side}"] = (ref_ts - ts).days
        else:
            ext[f"days_since_civ_{side}"] = None

        # ── Last 20 games covers P2 (MMR lags), P3 (recent form), P4 (avg_dur_20).
        #    Order is descending so index 0 = most recent game.
        #    mmr_lag{N} at inference = mmr[N-1] in this list (0-indexed):
        #      training LAG(mmr, N) at row G+1 = mmr at row (G+1-N) = mmr[G-(N-1)]
        #      which is the N-th element back from the most recent game → index N-1.
        recent_ts_clause = "AND g.started_at < ?" if before_timestamp is not None else ""
        recent_params = [pid] + ([before_timestamp] if before_timestamp is not None else [])
        recent = conn.execute(f"""
            SELECT p.mmr, p.result::INT, g.duration
            FROM participants p
            JOIN games g ON p.game_id = g.game_id
            WHERE p.profile_id = ?
              AND g.kind IN ('rm_1v1', 'rm_solo')
              AND p.result IS NOT NULL
              AND g.started_at IS NOT NULL
              {recent_ts_clause}
            ORDER BY g.started_at DESC, g.game_id DESC
            LIMIT 20
        """, recent_params).fetchall()

        # P2: MMR lags and rolling stats (keep nulls to match LAG behaviour)
        mmr_seq = [r[0] for r in recent]
        for n in (3, 5, 10, 20):
            ext[f"mmr_lag{n}_{side}"] = mmr_seq[n - 1] if len(mmr_seq) >= n else None
        mmrs_10 = [m for m in mmr_seq[:10] if m is not None]
        mmrs_20 = [m for m in mmr_seq[:20] if m is not None]
        ext[f"mmr_std_10_{side}"] = float(np.std(mmrs_10, ddof=1)) if len(mmrs_10) >= 2 else None
        ext[f"mmr_std_20_{side}"] = float(np.std(mmrs_20, ddof=1)) if len(mmrs_20) >= 2 else None

        # P3: recent form (results are always non-null — filtered above)
        results = [r[1] for r in recent]
        for n in (5, 10, 20):
            ext[f"recent_n_{n}_{side}"] = min(n, len(results))
            ext[f"recent_w_{n}_{side}"] = int(sum(results[:n]))

        # P4: avg_dur_20
        dur20 = [r[2] for r in recent if r[2] is not None]
        ext[f"avg_dur_20_{side}"] = float(np.mean(dur20)) if dur20 else None

    # ── P5: head-to-head record between the exact pair ────────────────────────
    h2h_ts_clause = "AND g.started_at < ?" if before_timestamp is not None else ""
    h2h_params = [pid_lo, pid_hi] + ([before_timestamp] if before_timestamp is not None else [])
    h2h = conn.execute(f"""
        SELECT COUNT(*), COALESCE(SUM(p_lo.result::INT), 0)
        FROM participants p_lo
        JOIN participants p_hi ON p_lo.game_id = p_hi.game_id
        JOIN games g ON p_lo.game_id = g.game_id
        WHERE p_lo.profile_id = ? AND p_hi.profile_id = ?
          AND g.kind IN ('rm_1v1', 'rm_solo')
          AND p_lo.result IS NOT NULL
          {h2h_ts_clause}
    """, h2h_params).fetchone()
    h2h_games    = h2h[0] or 0
    h2h_wins_lo  = h2h[1] or 0
    ext["h2h_games"]  = h2h_games
    ext["h2h_wins_a"] = h2h_wins_lo if lo_is_a else (h2h_games - h2h_wins_lo)

    # ── Apply Python-level derivations via existing _pN_derived functions ─────
    # P8 features are derived entirely from base_feat columns (no new queries).
    merged = {**base_feat, **ext}
    df = pd.DataFrame([merged])
    df = _p1_derived(df)
    df = _p2_derived(df)
    df = _p3_derived(df)
    df = _p4_derived(df)
    df = _p5_derived(df)
    df = _p8_derived(df)
    df = _p9_derived(df)

    row = df.iloc[0].to_dict()
    # Return only the keys that aren't already in base_feat (avoids double-writing)
    return {k: v for k, v in row.items() if k not in base_feat}


# ── capped-history training overrides for API-style experiments ──────────────

def build_api_cap_p1_p3_p4_p5_overrides(conn, visible_match_cap: int) -> pd.DataFrame:
    """
    Return one row per training_features game_id with P1/P3/P4/P5 semantics
    recomputed
    under a "last N prior games per player" visibility cap.

    This mirrors a recent-games API page at training time:
      - P1 civ-recency stats are limited to the last N visible prior games.
      - P3 recent-form semantics are unchanged because the baseline already uses
        only the most recent 5/10/20 games, which fit inside the capped slice.
      - P4 duration-profile stats are limited to the last N visible prior games.
      - P5 head-to-head only counts prior meetings visible in both players'
        capped N-game histories.

    training_features must already exist and contain the requested seasons.
    """
    if visible_match_cap <= 0:
        raise ValueError("visible_match_cap must be positive")
    query = """
    WITH ordered AS (
        SELECT
            ps.game_id,
            ps.profile_id,
            ps.result::INT AS result,
            ps.civ,
            ps.started_at,
            g.duration,
            ROW_NUMBER() OVER (
                PARTITION BY ps.profile_id
                ORDER BY ps.started_at, ps.game_id
            ) AS seq
        FROM player_stats ps
        JOIN games g ON ps.game_id = g.game_id
    ),
    side_a AS (
        SELECT
            tf.game_id,
            COUNT(prev.game_id) AS visible_games_a,
            AVG(prev.duration) AS avg_dur_life_a,
            AVG(prev.duration) FILTER (WHERE prev.civ = tf.civ_a) AS civ_avg_dur_a,
            AVG(prev.duration) FILTER (WHERE prev.seq >= cur.seq - 20) AS avg_dur_20_a,
            COUNT(prev.game_id) FILTER (
                WHERE prev.civ = tf.civ_a
                  AND prev.started_at >= tf.started_at - INTERVAL '7 days'
            ) AS civ_games_7d_a,
            COALESCE(SUM(prev.result) FILTER (
                WHERE prev.civ = tf.civ_a
                  AND prev.started_at >= tf.started_at - INTERVAL '7 days'
            ), 0) AS civ_wins_7d_a,
            COUNT(prev.game_id) FILTER (
                WHERE prev.civ = tf.civ_a
                  AND prev.started_at >= tf.started_at - INTERVAL '30 days'
            ) AS civ_games_30d_a,
            COALESCE(SUM(prev.result) FILTER (
                WHERE prev.civ = tf.civ_a
                  AND prev.started_at >= tf.started_at - INTERVAL '30 days'
            ), 0) AS civ_wins_30d_a,
            COUNT(prev.game_id) FILTER (
                WHERE prev.civ = tf.civ_a
                  AND prev.started_at >= tf.started_at - INTERVAL '60 days'
            ) AS civ_games_60d_a,
            COALESCE(SUM(prev.result) FILTER (
                WHERE prev.civ = tf.civ_a
                  AND prev.started_at >= tf.started_at - INTERVAL '60 days'
            ), 0) AS civ_wins_60d_a,
            DATEDIFF(
                'day',
                MAX(prev.started_at) FILTER (WHERE prev.civ = tf.civ_a),
                tf.started_at
            ) AS days_since_civ_a,
            COUNT(prev.game_id) FILTER (
                WHERE prev.started_at >= tf.started_at - INTERVAL '30 days'
            ) AS act_games_30d_a,
            COUNT(prev.game_id) FILTER (
                WHERE prev.started_at >= tf.started_at - INTERVAL '60 days'
            ) AS act_games_60d_a,
            COUNT(prev.game_id) FILTER (WHERE prev.duration IS NOT NULL AND prev.duration <= 900) AS short_games_a,
            COALESCE(SUM(prev.result) FILTER (
                WHERE prev.duration IS NOT NULL AND prev.duration <= 900
            ), 0) AS short_wins_a,
            COUNT(prev.game_id) FILTER (WHERE prev.duration IS NOT NULL AND prev.duration > 1800) AS long_games_a,
            COALESCE(SUM(prev.result) FILTER (
                WHERE prev.duration IS NOT NULL AND prev.duration > 1800
            ), 0) AS long_wins_a
        FROM training_features tf
        JOIN ordered cur
          ON tf.game_id = cur.game_id
         AND tf.profile_id_a = cur.profile_id
        LEFT JOIN ordered prev
          ON prev.profile_id = tf.profile_id_a
         AND prev.seq BETWEEN cur.seq - $visible_match_cap AND cur.seq - 1
        GROUP BY tf.game_id, tf.started_at, tf.civ_a, cur.seq
    ),
    side_b AS (
        SELECT
            tf.game_id,
            COUNT(prev.game_id) AS visible_games_b,
            AVG(prev.duration) AS avg_dur_life_b,
            AVG(prev.duration) FILTER (WHERE prev.civ = tf.civ_b) AS civ_avg_dur_b,
            AVG(prev.duration) FILTER (WHERE prev.seq >= cur.seq - 20) AS avg_dur_20_b,
            COUNT(prev.game_id) FILTER (
                WHERE prev.civ = tf.civ_b
                  AND prev.started_at >= tf.started_at - INTERVAL '7 days'
            ) AS civ_games_7d_b,
            COALESCE(SUM(prev.result) FILTER (
                WHERE prev.civ = tf.civ_b
                  AND prev.started_at >= tf.started_at - INTERVAL '7 days'
            ), 0) AS civ_wins_7d_b,
            COUNT(prev.game_id) FILTER (
                WHERE prev.civ = tf.civ_b
                  AND prev.started_at >= tf.started_at - INTERVAL '30 days'
            ) AS civ_games_30d_b,
            COALESCE(SUM(prev.result) FILTER (
                WHERE prev.civ = tf.civ_b
                  AND prev.started_at >= tf.started_at - INTERVAL '30 days'
            ), 0) AS civ_wins_30d_b,
            COUNT(prev.game_id) FILTER (
                WHERE prev.civ = tf.civ_b
                  AND prev.started_at >= tf.started_at - INTERVAL '60 days'
            ) AS civ_games_60d_b,
            COALESCE(SUM(prev.result) FILTER (
                WHERE prev.civ = tf.civ_b
                  AND prev.started_at >= tf.started_at - INTERVAL '60 days'
            ), 0) AS civ_wins_60d_b,
            DATEDIFF(
                'day',
                MAX(prev.started_at) FILTER (WHERE prev.civ = tf.civ_b),
                tf.started_at
            ) AS days_since_civ_b,
            COUNT(prev.game_id) FILTER (
                WHERE prev.started_at >= tf.started_at - INTERVAL '30 days'
            ) AS act_games_30d_b,
            COUNT(prev.game_id) FILTER (
                WHERE prev.started_at >= tf.started_at - INTERVAL '60 days'
            ) AS act_games_60d_b,
            COUNT(prev.game_id) FILTER (WHERE prev.duration IS NOT NULL AND prev.duration <= 900) AS short_games_b,
            COALESCE(SUM(prev.result) FILTER (
                WHERE prev.duration IS NOT NULL AND prev.duration <= 900
            ), 0) AS short_wins_b,
            COUNT(prev.game_id) FILTER (WHERE prev.duration IS NOT NULL AND prev.duration > 1800) AS long_games_b,
            COALESCE(SUM(prev.result) FILTER (
                WHERE prev.duration IS NOT NULL AND prev.duration > 1800
            ), 0) AS long_wins_b
        FROM training_features tf
        JOIN ordered cur
          ON tf.game_id = cur.game_id
         AND tf.profile_id_b = cur.profile_id
        LEFT JOIN ordered prev
          ON prev.profile_id = tf.profile_id_b
         AND prev.seq BETWEEN cur.seq - $visible_match_cap AND cur.seq - 1
        GROUP BY tf.game_id, tf.started_at, tf.civ_b, cur.seq
    ),
    h2h AS (
        SELECT
            tf.game_id,
            COUNT(pb.game_id) AS h2h_games,
            COALESCE(SUM(pa.result) FILTER (WHERE pb.game_id IS NOT NULL), 0) AS h2h_wins_a
        FROM training_features tf
        JOIN ordered cur_a
          ON tf.game_id = cur_a.game_id
         AND tf.profile_id_a = cur_a.profile_id
        JOIN ordered cur_b
          ON tf.game_id = cur_b.game_id
         AND tf.profile_id_b = cur_b.profile_id
        LEFT JOIN ordered pa
          ON pa.profile_id = tf.profile_id_a
         AND pa.seq BETWEEN cur_a.seq - $visible_match_cap AND cur_a.seq - 1
        LEFT JOIN ordered pb
          ON pb.profile_id = tf.profile_id_b
         AND pb.game_id = pa.game_id
         AND pb.seq BETWEEN cur_b.seq - $visible_match_cap AND cur_b.seq - 1
        GROUP BY tf.game_id
    )
    SELECT
        tf.game_id,
        side_a.* EXCLUDE (game_id),
        side_b.* EXCLUDE (game_id),
        h2h.h2h_games,
        h2h.h2h_wins_a
    FROM training_features tf
    LEFT JOIN side_a ON tf.game_id = side_a.game_id
    LEFT JOIN side_b ON tf.game_id = side_b.game_id
    LEFT JOIN h2h ON tf.game_id = h2h.game_id
    ORDER BY tf.game_id
    """
    return conn.execute(query, {"visible_match_cap": visible_match_cap}).df()


def build_api50_p1_p3_p4_p5_overrides(conn) -> pd.DataFrame:
    return build_api_cap_p1_p3_p4_p5_overrides(conn, visible_match_cap=50)


def apply_api50_p1_p3_p4_p5_overrides(df: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    """
    Replace full-history P1/P3/P4/P5 feature values with their API-style capped
    last-50-games equivalents.

    Only P1/P4/P5 columns are overwritten. P3 columns are intentionally left
    unchanged because their 5/10/20-game windows are already fully recoverable
    inside the capped 50-game visible slice.
    """
    merged = df.merge(overrides, on="game_id", how="left", suffixes=("", "__api50"))

    def assign(col: str) -> None:
        api_col = f"{col}__api50"
        if api_col not in merged.columns:
            return
        merged[col] = merged[api_col]

    raw_cols = [
        "visible_games_a", "visible_games_b",
        "avg_dur_life_a", "avg_dur_life_b",
        "avg_dur_20_a", "avg_dur_20_b",
        "civ_avg_dur_a", "civ_avg_dur_b",
        "civ_games_7d_a", "civ_wins_7d_a", "civ_games_30d_a", "civ_wins_30d_a",
        "civ_games_60d_a", "civ_wins_60d_a", "days_since_civ_a",
        "act_games_30d_a", "act_games_60d_a",
        "short_games_a", "short_wins_a", "long_games_a", "long_wins_a",
        "civ_games_7d_b", "civ_wins_7d_b", "civ_games_30d_b", "civ_wins_30d_b",
        "civ_games_60d_b", "civ_wins_60d_b", "days_since_civ_b",
        "act_games_30d_b", "act_games_60d_b",
        "short_games_b", "short_wins_b", "long_games_b", "long_wins_b",
        "h2h_games", "h2h_wins_a",
    ]
    for col in raw_cols:
        assign(col)

    for side in ("a", "b"):
        merged[f"civ_wr_7d_{side}"] = _smooth(
            merged[f"civ_wins_7d_{side}"].fillna(0),
            merged[f"civ_games_7d_{side}"].fillna(0),
        )
        merged[f"civ_wr_30d_{side}"] = _smooth(
            merged[f"civ_wins_30d_{side}"].fillna(0),
            merged[f"civ_games_30d_{side}"].fillna(0),
        )
        merged[f"civ_wr_60d_{side}"] = _smooth(
            merged[f"civ_wins_60d_{side}"].fillna(0),
            merged[f"civ_games_60d_{side}"].fillna(0),
        )
        merged[f"civ_frac_30d_{side}"] = (
            merged[f"civ_games_30d_{side}"].fillna(0)
            / merged[f"act_games_30d_{side}"].fillna(0).clip(lower=1)
        )
        merged[f"civ_frac_60d_{side}"] = (
            merged[f"civ_games_60d_{side}"].fillna(0)
            / merged[f"act_games_60d_{side}"].fillna(0).clip(lower=1)
        )
        merged[f"short_wr_{side}"] = _smooth(
            merged[f"short_wins_{side}"].fillna(0),
            merged[f"short_games_{side}"].fillna(0),
        )
        merged[f"long_wr_{side}"] = _smooth(
            merged[f"long_wins_{side}"].fillna(0),
            merged[f"long_games_{side}"].fillna(0),
        )
        visible = merged[f"visible_games_{side}"].fillna(0).clip(lower=1)
        merged[f"short_share_{side}"] = merged[f"short_games_{side}"].fillna(0) / visible
        merged[f"long_share_{side}"] = merged[f"long_games_{side}"].fillna(0) / visible

    merged["civ_wr_30d_diff"] = merged["civ_wr_30d_a"] - merged["civ_wr_30d_b"]
    merged["civ_wr_60d_diff"] = merged["civ_wr_60d_a"] - merged["civ_wr_60d_b"]
    merged["civ_frac_30d_diff"] = merged["civ_frac_30d_a"] - merged["civ_frac_30d_b"]
    merged["civ_frac_60d_diff"] = merged["civ_frac_60d_a"] - merged["civ_frac_60d_b"]

    merged["avg_dur_life_diff"] = merged["avg_dur_life_a"].fillna(0) - merged["avg_dur_life_b"].fillna(0)
    merged["avg_dur_20_diff"] = merged["avg_dur_20_a"].fillna(0) - merged["avg_dur_20_b"].fillna(0)
    merged["civ_avg_dur_diff"] = merged["civ_avg_dur_a"].fillna(0) - merged["civ_avg_dur_b"].fillna(0)
    merged["short_wr_diff"] = merged["short_wr_a"] - merged["short_wr_b"]
    merged["long_wr_diff"] = merged["long_wr_a"] - merged["long_wr_b"]
    merged["short_share_diff"] = merged["short_share_a"] - merged["short_share_b"]
    merged["long_share_diff"] = merged["long_share_a"] - merged["long_share_b"]

    merged["h2h_wr_a"] = (merged["h2h_wins_a"].fillna(0) + 2.5) / (merged["h2h_games"].fillna(0) + 5)
    if "h2h_has_data" in merged.columns:
        merged["h2h_has_data"] = (merged["h2h_games"].fillna(0) >= 3).astype(int)

    drop_cols = [c for c in merged.columns if c.endswith("__api50")]
    drop_cols.extend(
        c for c in [
            "visible_games_a", "visible_games_b",
            "act_games_30d_a", "act_games_60d_a",
            "act_games_30d_b", "act_games_60d_b",
        ]
        if c in merged.columns
    )
    if drop_cols:
        merged = merged.drop(columns=drop_cols)
    return merged


def apply_api_cap_p1_p3_p4_p5_overrides(df: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    return apply_api50_p1_p3_p4_p5_overrides(df, overrides)


# Backward-compatible aliases while the experiment entrypoint settles on the
# P1/P3/P4/P5 naming used in the comparison spec.
def build_api50_p1_p4_p5_overrides(conn) -> pd.DataFrame:
    return build_api50_p1_p3_p4_p5_overrides(conn)


def apply_api50_p1_p4_p5_overrides(df: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    return apply_api50_p1_p3_p4_p5_overrides(df, overrides)


# ── recent-only base overrides + full-history career block ───────────────────
#
# These power the "historic integration" experiment
# (scripts/experiments/compare_historic_integration.py).
#
#   build_recent_only_base_overrides() caps the BASE history-derived counts
#   (lifetime / season / civ / map games+wins) to each player's last N prior
#   games — an honest "one aoe4world page" mock. Combined with the existing
#   build_api_cap_p1_p3_p4_p5_overrides(), the whole feature set then reflects
#   only the recent window (MMR/rating are retained — the API returns them
#   per game).
#
#   build_career_block() adds full-history career summaries as SEPARATE columns
#   (peak/avg MMR, career WR, civ proficiency, form-vs-career), standing in for
#   a periodically-refreshed career-summary cache. These complement, rather than
#   replace, the recent-window signal.

# Base raw count columns recomputed under the recent-window cap.
_RECENT_ONLY_BASE_COLS = [
    "games_lifetime_a", "wins_lifetime_a",
    "games_season_a", "wins_season_a",
    "civ_games_a", "civ_wins_a",
    "map_games_a", "map_wins_a",
    "games_lifetime_b", "wins_lifetime_b",
    "games_season_b", "wins_season_b",
    "civ_games_b", "civ_wins_b",
    "map_games_b", "map_wins_b",
]


def build_recent_only_base_overrides(conn, visible_match_cap: int) -> pd.DataFrame:
    """
    Return one row per training_features game_id with the BASE history counts
    (lifetime / season / civ / map games+wins) recomputed from only each
    player's last `visible_match_cap` prior games.

    Mirrors the last-N visibility of a single aoe4world recent-games page, the
    same window semantics used by build_api_cap_p1_p3_p4_p5_overrides. The
    `days_since_*` features need no override (the most recent prior game is
    always inside the window). MMR/rating are likewise untouched.

    training_features and player_stats must already exist.
    """
    if visible_match_cap <= 0:
        raise ValueError("visible_match_cap must be positive")
    query = """
    WITH ordered AS (
        SELECT
            ps.game_id,
            ps.profile_id,
            ps.result::INT AS result,
            ps.civ,
            ps.map,
            ps.season,
            ROW_NUMBER() OVER (
                PARTITION BY ps.profile_id
                ORDER BY ps.started_at, ps.game_id
            ) AS seq
        FROM player_stats ps
    ),
    side_a AS (
        SELECT
            tf.game_id,
            COUNT(prev.game_id) AS games_lifetime_a,
            COALESCE(SUM(prev.result), 0) AS wins_lifetime_a,
            COUNT(prev.game_id) FILTER (WHERE prev.season = tf.season) AS games_season_a,
            COALESCE(SUM(prev.result) FILTER (WHERE prev.season = tf.season), 0) AS wins_season_a,
            COUNT(prev.game_id) FILTER (WHERE prev.civ = tf.civ_a) AS civ_games_a,
            COALESCE(SUM(prev.result) FILTER (WHERE prev.civ = tf.civ_a), 0) AS civ_wins_a,
            COUNT(prev.game_id) FILTER (WHERE prev.map = tf.map) AS map_games_a,
            COALESCE(SUM(prev.result) FILTER (WHERE prev.map = tf.map), 0) AS map_wins_a
        FROM training_features tf
        JOIN ordered cur
          ON tf.game_id = cur.game_id
         AND tf.profile_id_a = cur.profile_id
        LEFT JOIN ordered prev
          ON prev.profile_id = tf.profile_id_a
         AND prev.seq BETWEEN cur.seq - $visible_match_cap AND cur.seq - 1
        GROUP BY tf.game_id, cur.seq
    ),
    side_b AS (
        SELECT
            tf.game_id,
            COUNT(prev.game_id) AS games_lifetime_b,
            COALESCE(SUM(prev.result), 0) AS wins_lifetime_b,
            COUNT(prev.game_id) FILTER (WHERE prev.season = tf.season) AS games_season_b,
            COALESCE(SUM(prev.result) FILTER (WHERE prev.season = tf.season), 0) AS wins_season_b,
            COUNT(prev.game_id) FILTER (WHERE prev.civ = tf.civ_b) AS civ_games_b,
            COALESCE(SUM(prev.result) FILTER (WHERE prev.civ = tf.civ_b), 0) AS civ_wins_b,
            COUNT(prev.game_id) FILTER (WHERE prev.map = tf.map) AS map_games_b,
            COALESCE(SUM(prev.result) FILTER (WHERE prev.map = tf.map), 0) AS map_wins_b
        FROM training_features tf
        JOIN ordered cur
          ON tf.game_id = cur.game_id
         AND tf.profile_id_b = cur.profile_id
        LEFT JOIN ordered prev
          ON prev.profile_id = tf.profile_id_b
         AND prev.seq BETWEEN cur.seq - $visible_match_cap AND cur.seq - 1
        GROUP BY tf.game_id, cur.seq
    )
    SELECT
        tf.game_id,
        side_a.* EXCLUDE (game_id),
        side_b.* EXCLUDE (game_id)
    FROM training_features tf
    LEFT JOIN side_a ON tf.game_id = side_a.game_id
    LEFT JOIN side_b ON tf.game_id = side_b.game_id
    ORDER BY tf.game_id
    """
    return conn.execute(query, {"visible_match_cap": visible_match_cap}).df()


def apply_recent_only_base_overrides(df: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    """
    Overwrite the base history-count columns with their recent-window values and
    recompute the base derived features (overall_wr, season_wr, civ_wr, map_wr,
    games_diff, wr_diff, new-player flags) from the capped counts.
    """
    from .features import _add_derived_features

    merged = df.merge(overrides, on="game_id", how="left", suffixes=("", "__cap"))
    for col in _RECENT_ONLY_BASE_COLS:
        cap_col = f"{col}__cap"
        if cap_col in merged.columns:
            merged[col] = merged[cap_col].fillna(0)
    drop_cols = [c for c in merged.columns if c.endswith("__cap")]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)
    # Regenerate base derived features from the now-capped raw counts.
    merged = _add_derived_features(merged)
    return merged


# Raw career columns emitted by build_career_block (joined, not overwritten).
_CAREER_RAW_COLS = [
    "career_games_a", "career_wins_a", "career_civ_games_a", "career_civ_wins_a",
    "peak_mmr_a", "career_avg_mmr_a",
    "career_games_b", "career_wins_b", "career_civ_games_b", "career_civ_wins_b",
    "peak_mmr_b", "career_avg_mmr_b",
]


def build_career_block(conn) -> pd.DataFrame:
    """
    Return one row per training_features game_id with FULL-HISTORY career
    summaries for both players, computed leakage-free (prior games only).

    These survive the recent-window cap and stand in for a stored, periodically
    refreshed career-summary cache:
      - career_games / career_wins  → smoothed career win rate
      - career_civ_games / career_civ_wins → smoothed career civ win rate
      - peak_mmr   = max prior MMR (skill ceiling)
      - career_avg_mmr = mean prior MMR (skill norm)

    player_stats already carries leakage-safe lifetime/civ cumulative counts, so
    only peak/avg MMR need fresh window functions.
    """
    query = """
    WITH career AS (
        SELECT
            game_id,
            profile_id,
            games_lifetime_before AS career_games,
            wins_lifetime_before  AS career_wins,
            civ_games_before      AS career_civ_games,
            civ_wins_before       AS career_civ_wins,
            MAX(mmr) OVER w AS peak_mmr,
            AVG(mmr) OVER w AS career_avg_mmr
        FROM player_stats
        WINDOW w AS (
            PARTITION BY profile_id
            ORDER BY started_at, game_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )
    )
    SELECT
        tf.game_id,
        ca.career_games     AS career_games_a,
        ca.career_wins      AS career_wins_a,
        ca.career_civ_games AS career_civ_games_a,
        ca.career_civ_wins  AS career_civ_wins_a,
        ca.peak_mmr         AS peak_mmr_a,
        ca.career_avg_mmr   AS career_avg_mmr_a,
        cb.career_games     AS career_games_b,
        cb.career_wins      AS career_wins_b,
        cb.career_civ_games AS career_civ_games_b,
        cb.career_civ_wins  AS career_civ_wins_b,
        cb.peak_mmr         AS peak_mmr_b,
        cb.career_avg_mmr   AS career_avg_mmr_b
    FROM training_features tf
    LEFT JOIN career ca ON ca.game_id = tf.game_id AND ca.profile_id = tf.profile_id_a
    LEFT JOIN career cb ON cb.game_id = tf.game_id AND cb.profile_id = tf.profile_id_b
    ORDER BY tf.game_id
    """
    return conn.execute(query).df()


def apply_career_block(df: pd.DataFrame, career: pd.DataFrame) -> pd.DataFrame:
    """
    Merge full-history career columns and derive the modelled CAREER_FEATURES:
    smoothed career WR, civ WR, current-MMR-vs-peak/norm gaps, recent-form-minus-
    career delta, and cross-player diffs.

    Requires P3 `recent_wr_20_{a,b}` to already be present (adjusted_form family)
    for the form-vs-career deltas.
    """
    merged = df.merge(career, on="game_id", how="left")

    for side in ("a", "b"):
        merged[f"career_wr_{side}"] = _smooth(
            merged[f"career_wins_{side}"].fillna(0),
            merged[f"career_games_{side}"].fillna(0),
        )
        merged[f"career_civ_wr_{side}"] = _smooth(
            merged[f"career_civ_wins_{side}"].fillna(0),
            merged[f"career_civ_games_{side}"].fillna(0),
        )
        merged[f"mmr_vs_peak_{side}"] = merged[f"mmr_{side}"] - merged[f"peak_mmr_{side}"]
        merged[f"mmr_vs_career_avg_{side}"] = merged[f"mmr_{side}"] - merged[f"career_avg_mmr_{side}"]
        if f"recent_wr_20_{side}" in merged.columns:
            merged[f"form_vs_career_{side}"] = (
                merged[f"recent_wr_20_{side}"] - merged[f"career_wr_{side}"]
            )

    merged["career_wr_diff"] = merged["career_wr_a"] - merged["career_wr_b"]
    merged["career_games_diff"] = merged["career_games_a"].fillna(0) - merged["career_games_b"].fillna(0)
    merged["peak_mmr_diff"] = merged["peak_mmr_a"] - merged["peak_mmr_b"]
    merged["career_avg_mmr_diff"] = merged["career_avg_mmr_a"] - merged["career_avg_mmr_b"]

    # Drop the intermediate raw win counts; keep career_games (experience signal).
    drop_cols = [
        "career_wins_a", "career_civ_games_a", "career_civ_wins_a",
        "career_wins_b", "career_civ_games_b", "career_civ_wins_b",
    ]
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])
    return merged


# ── window-temporal block (computable from the aoe4world recent-games page) ──
#
# Every feature here is recoverable from ONLY the last N games returned by the
# aoe4world API (each game carries started_at), so no DB career cache is needed.
# Captures HOW the recent window is distributed in calendar time — a player who
# played their last 30 games in 3 days (hot/grinding) looks identical to one who
# took 8 months (rusty) on every existing recency feature except `days_since`.

def build_window_temporal_overrides(conn, visible_match_cap: int) -> pd.DataFrame:
    """
    Return one row per training_features game_id with calendar-time descriptors
    of each player's last `visible_match_cap` prior games:
      - wt_visible          : visible game count (helper; dropped after derive)
      - window_span_days    : days from the oldest visible game to the current match
      - gap_mean_window     : mean days between consecutive games inside the window
      - gap_max_window      : longest idle gap inside the window (return-from-break)
      - wt_act_7d / wt_act_30d : games played in the last 7 / 30 calendar days

    Uses player_stats.days_since_last_game (precomputed gap-since-previous), so the
    per-game gaps are a cheap aggregate rather than a correlated subquery.
    """
    if visible_match_cap <= 0:
        raise ValueError("visible_match_cap must be positive")
    query = """
    WITH ordered AS (
        SELECT
            ps.game_id,
            ps.profile_id,
            ps.started_at,
            ps.days_since_last_game,
            ROW_NUMBER() OVER (
                PARTITION BY ps.profile_id
                ORDER BY ps.started_at, ps.game_id
            ) AS seq
        FROM player_stats ps
    ),
    side_a AS (
        SELECT
            tf.game_id,
            COUNT(prev.game_id) AS wt_visible_a,
            DATEDIFF('day', MIN(prev.started_at), tf.started_at) AS window_span_days_a,
            AVG(prev.days_since_last_game) AS gap_mean_window_a,
            MAX(prev.days_since_last_game) AS gap_max_window_a,
            COUNT(prev.game_id) FILTER (
                WHERE prev.started_at >= tf.started_at - INTERVAL '7 days'
            ) AS wt_act_7d_a,
            COUNT(prev.game_id) FILTER (
                WHERE prev.started_at >= tf.started_at - INTERVAL '30 days'
            ) AS wt_act_30d_a
        FROM training_features tf
        JOIN ordered cur
          ON tf.game_id = cur.game_id
         AND tf.profile_id_a = cur.profile_id
        LEFT JOIN ordered prev
          ON prev.profile_id = tf.profile_id_a
         AND prev.seq BETWEEN cur.seq - $visible_match_cap AND cur.seq - 1
        GROUP BY tf.game_id, tf.started_at, cur.seq
    ),
    side_b AS (
        SELECT
            tf.game_id,
            COUNT(prev.game_id) AS wt_visible_b,
            DATEDIFF('day', MIN(prev.started_at), tf.started_at) AS window_span_days_b,
            AVG(prev.days_since_last_game) AS gap_mean_window_b,
            MAX(prev.days_since_last_game) AS gap_max_window_b,
            COUNT(prev.game_id) FILTER (
                WHERE prev.started_at >= tf.started_at - INTERVAL '7 days'
            ) AS wt_act_7d_b,
            COUNT(prev.game_id) FILTER (
                WHERE prev.started_at >= tf.started_at - INTERVAL '30 days'
            ) AS wt_act_30d_b
        FROM training_features tf
        JOIN ordered cur
          ON tf.game_id = cur.game_id
         AND tf.profile_id_b = cur.profile_id
        LEFT JOIN ordered prev
          ON prev.profile_id = tf.profile_id_b
         AND prev.seq BETWEEN cur.seq - $visible_match_cap AND cur.seq - 1
        GROUP BY tf.game_id, tf.started_at, cur.seq
    )
    SELECT
        tf.game_id,
        side_a.* EXCLUDE (game_id),
        side_b.* EXCLUDE (game_id)
    FROM training_features tf
    LEFT JOIN side_a ON tf.game_id = side_a.game_id
    LEFT JOIN side_b ON tf.game_id = side_b.game_id
    ORDER BY tf.game_id
    """
    return conn.execute(query, {"visible_match_cap": visible_match_cap}).df()


def apply_window_temporal_overrides(df: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    """Merge window-temporal columns, derive density and cross-player diffs."""
    merged = df.merge(overrides, on="game_id", how="left")
    for side in ("a", "b"):
        span = merged[f"window_span_days_{side}"].fillna(0).clip(lower=0)
        visible = merged[f"wt_visible_{side}"].fillna(0)
        merged[f"games_per_day_window_{side}"] = visible / (span + 1.0)

    merged["window_span_days_diff"] = (
        merged["window_span_days_a"].fillna(0) - merged["window_span_days_b"].fillna(0)
    )
    merged["games_per_day_window_diff"] = (
        merged["games_per_day_window_a"] - merged["games_per_day_window_b"]
    )
    merged["gap_max_window_diff"] = (
        merged["gap_max_window_a"].fillna(0) - merged["gap_max_window_b"].fillna(0)
    )
    merged["wt_act_30d_diff"] = (
        merged["wt_act_30d_a"].fillna(0) - merged["wt_act_30d_b"].fillna(0)
    )

    merged = merged.drop(columns=[c for c in ("wt_visible_a", "wt_visible_b") if c in merged.columns])
    return merged


# ── feature name lists (for model.py and ablation) ───────────────────────────

P1_FEATURES = [
    "civ_games_7d_a", "civ_wins_7d_a", "civ_games_30d_a", "civ_wins_30d_a",
    "civ_games_60d_a", "civ_wins_60d_a", "days_since_civ_a",
    "civ_wr_7d_a", "civ_wr_30d_a", "civ_wr_60d_a",
    "civ_frac_30d_a", "civ_frac_60d_a",
    "civ_games_7d_b", "civ_wins_7d_b", "civ_games_30d_b", "civ_wins_30d_b",
    "civ_games_60d_b", "civ_wins_60d_b", "days_since_civ_b",
    "civ_wr_7d_b", "civ_wr_30d_b", "civ_wr_60d_b",
    "civ_frac_30d_b", "civ_frac_60d_b",
    "civ_wr_30d_diff", "civ_wr_60d_diff", "civ_frac_30d_diff", "civ_frac_60d_diff",
]

P2_FEATURES = [
    "mmr_lag3_a", "mmr_lag5_a", "mmr_lag10_a", "mmr_lag20_a",
    "mmr_std_10_a", "mmr_std_20_a",
    "mmr_change_3_a", "mmr_change_5_a", "mmr_change_10_a", "mmr_change_20_a",
    "mmr_slope_10_a", "mmr_slope_20_a", "mmr_rising_10_a",
    "mmr_lag3_b", "mmr_lag5_b", "mmr_lag10_b", "mmr_lag20_b",
    "mmr_std_10_b", "mmr_std_20_b",
    "mmr_change_3_b", "mmr_change_10_b", "mmr_change_20_b",
    "mmr_slope_10_b", "mmr_slope_20_b", "mmr_rising_10_b",
    "mmr_change_10_diff", "mmr_change_20_diff",
    "mmr_slope_10_diff", "mmr_std_10_diff", "mmr_std_20_diff",
]

P3_FEATURES = [
    "recent_w_5_a", "recent_wr_5_a",
    "recent_w_10_a", "recent_wr_10_a",
    "recent_w_20_a", "recent_wr_20_a",
    "recent_w_5_b", "recent_wr_5_b",
    "recent_w_10_b", "recent_wr_10_b",
    "recent_w_20_b", "recent_wr_20_b",
    "recent_wr_5_diff", "recent_wr_10_diff", "recent_wr_20_diff",
]

P4_FEATURES = [
    "avg_dur_life_a", "avg_dur_20_a", "civ_avg_dur_a",
    "short_games_a", "short_wins_a", "short_wr_a", "short_share_a",
    "long_games_a", "long_wins_a", "long_wr_a", "long_share_a",
    "avg_dur_life_b", "avg_dur_20_b", "civ_avg_dur_b",
    "short_games_b", "short_wins_b", "short_wr_b", "short_share_b",
    "long_games_b", "long_wins_b", "long_wr_b", "long_share_b",
    "avg_dur_life_diff", "avg_dur_20_diff", "civ_avg_dur_diff",
    "short_wr_diff", "long_wr_diff", "short_share_diff", "long_share_diff",
]

P5_FEATURES = [
    "h2h_games", "h2h_wins_a", "h2h_wr_a",
]

P8_FEATURES = [
    "both_mmr_missing", "a_mmr_missing_only", "b_mmr_missing_only",
    "a_low_lt5", "a_low_lt10", "a_low_lt20",
    "b_low_lt5", "b_low_lt10", "b_low_lt20",
    "one_low_one_est", "both_low",
    "is_first10_season_a", "is_first10_season_b",
]

P9_FEATURES = [
    "act_games_7d_a", "act_games_14d_a", "act_games_30d_a", "act_games_60d_a",
    "act_games_7d_b", "act_games_14d_b", "act_games_30d_b", "act_games_60d_b",
    "inactive_7d_a", "inactive_30d_a",
    "inactive_7d_b", "inactive_30d_b",
    "act_games_7d_diff", "act_games_14d_diff", "act_games_30d_diff",
]

P6_FEATURES = [
    # map_metadata_missing: numeric flag (1 = no metadata for this map)
    "map_metadata_missing",
    # Player archetype history counts and smoothed WR
    # (map taxonomy categoricals are in model.py CATEGORICAL_FEATURES, not duplicated here)
    "map_type_games_a", "map_type_wins_a", "map_type_wr_a",
    "map_type_games_b", "map_type_wins_b", "map_type_wr_b",
    "openness_games_a", "openness_wins_a", "openness_wr_a",
    "openness_games_b", "openness_wins_b", "openness_wr_b",
    "map_type_wr_diff", "openness_wr_diff",
]

P7_FEATURES = [
    "prior_map_games",
    "log_prior_map_games",
    "prior_avg_duration",
    "prior_short_game_share",
    "prior_long_game_share",
    # Patch-age features (only populated when patch_metadata is available)
    "patch_age_days",
    "is_first_7d_of_patch",
    "is_major_patch",
    "is_balance_patch",
    "missing_patch_start",
]

# Full-history career-summary block (historic-integration experiment). Added to
# the matrix only by apply_career_block(); absent from normal training runs, so
# the trainer's `[c for c in ALL_FEATURES if c in df.columns]` filter skips them.
CAREER_FEATURES = [
    "career_games_a", "career_wr_a", "career_civ_wr_a",
    "peak_mmr_a", "career_avg_mmr_a",
    "mmr_vs_peak_a", "mmr_vs_career_avg_a", "form_vs_career_a",
    "career_games_b", "career_wr_b", "career_civ_wr_b",
    "peak_mmr_b", "career_avg_mmr_b",
    "mmr_vs_peak_b", "mmr_vs_career_avg_b", "form_vs_career_b",
    "career_wr_diff", "career_games_diff", "peak_mmr_diff", "career_avg_mmr_diff",
]

# Calendar-time descriptors of the recent window — recoverable from the
# aoe4world recent-games page alone (no DB cache). Added only by
# apply_window_temporal_overrides(); absent from normal runs (selector skips them).
WINDOW_TEMPORAL_FEATURES = [
    "window_span_days_a", "gap_mean_window_a", "gap_max_window_a",
    "wt_act_7d_a", "wt_act_30d_a", "games_per_day_window_a",
    "window_span_days_b", "gap_mean_window_b", "gap_max_window_b",
    "wt_act_7d_b", "wt_act_30d_b", "games_per_day_window_b",
    "window_span_days_diff", "games_per_day_window_diff",
    "gap_max_window_diff", "wt_act_30d_diff",
]

ALL_EXTRA_FEATURES = (
    P1_FEATURES + P2_FEATURES + P3_FEATURES + P4_FEATURES +
    P5_FEATURES + P6_FEATURES + P7_FEATURES + P8_FEATURES + P9_FEATURES +
    CAREER_FEATURES + WINDOW_TEMPORAL_FEATURES
)

FAMILY_FEATURES: dict[str, list[str]] = {
    "civ_recency":        P1_FEATURES,
    "mmr_trend":          P2_FEATURES,
    "adjusted_form":      P3_FEATURES,
    "duration_profile":   P4_FEATURES,
    "head_to_head":       P5_FEATURES,
    "map_archetypes":     P6_FEATURES,
    "patch_priors":       P7_FEATURES,
    "low_history_detail": P8_FEATURES,
    "activity_session":   P9_FEATURES,
}

# Families that showed no improvement on S10+S11 ablation (Brier Δ < 0.001).
# Excluded from --add-all-families but still callable explicitly.
DISABLED_FAMILIES: frozenset[str] = frozenset({
    "map_archetypes", "patch_priors",          # P6/P7: negligible lift
    "mmr_trend", "low_history_detail", "activity_session",  # P2/P8/P9: zero or negative marginal value
})

#: New map taxonomy columns that need LightGBM categorical treatment
P6_CATEGORICAL_FEATURES = [
    "map_type_primary",
    "map_geography_subtype",
    "water_importance",
    "openness",
    "chokepoint_level",
    "trade_viability",
    "resource_scarcity",
    "map_metadata_confidence",
]
