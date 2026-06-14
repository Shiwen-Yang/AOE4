"""
Team-match feature engineering.

Pipeline (all leakage-free — every historical stat uses only games strictly before
the match being scored):

  1. player_stats        — per (team game, player) cumulative team-ladder history
                           (lifetime games/wins, last-10 form, civ familiarity, recency).
  2. onev1_snap          — per (team game, player) snapshot of that player's 1v1-ladder
                           skill AS OF the moment before the team match, via an ASOF join
                           into the READ-ONLY attached 1v1 database.
  3. team_agg            — one row per (game_id, team_id): MMR aggregates, the boost/carry
                           dispersion family, history & civ means, and 1v1 cross-reference.
  4. build_training_matrix → one row per MATCH (Team A = lower min profile_id), with A-vs-B
                           diffs and the binary target (1 = Team A won).

Skill convention: `skill = COALESCE(mmr, rating)` so dispersion features survive the ~few
percent of players missing an MMR; MMR-specific aggregates use `mmr` directly (matchmaking
balances on MMR).
"""
import time

import numpy as np
import pandas as pd

from .config import (
    CARRY_STD_K,
    GLOBAL_WR_PRIOR,
    LOW_MMR_FLOOR,
    NEW_PLAYER_THRESHOLD,
    PRIOR_STRENGTH,
)
from .db import attach_1v1, get_conn, table_exists

# Smurf signal: a player whose 1v1 MMR exceeds their team MMR by this much is a
# likely booster/smurf riding a deflated team rating.
SMURF_GAP = 200


def _season_list_sql(seasons: list[int]) -> str:
    return "(" + ",".join(str(int(s)) for s in seasons) + ")"


def build_player_stats(conn, mode: str, seasons: list[int]) -> None:
    """Per (team game, player) leakage-free team-ladder history."""
    sql = f"""
    CREATE OR REPLACE TABLE player_stats AS
    WITH player_game AS (
        SELECT
            p.game_id,
            p.profile_id,
            p.team_id,
            p.result::INT          AS result,
            p.civilization         AS civ,
            p.civilization_randomized,
            p.rating,
            p.mmr,
            COALESCE(p.mmr, p.rating) AS skill,
            g.started_at
        FROM participants p
        JOIN games g ON p.game_id = g.game_id
        WHERE g.kind = '{mode}'
          AND g.season IN {_season_list_sql(seasons)}
          AND p.result IS NOT NULL
          AND g.started_at IS NOT NULL
    )
    SELECT
        game_id, profile_id, team_id, result, civ, civilization_randomized,
        rating, mmr, skill, started_at,

        (ROW_NUMBER() OVER w_life - 1)                                  AS games_lifetime_before,
        COALESCE(SUM(result) OVER (PARTITION BY profile_id
            ORDER BY started_at, game_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0)       AS wins_lifetime_before,

        COALESCE(SUM(result) OVER (PARTITION BY profile_id
            ORDER BY started_at, game_id
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING), 0)             AS wins_last10_before,
        COALESCE(COUNT(*) OVER (PARTITION BY profile_id
            ORDER BY started_at, game_id
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING), 0)            AS n_last10_before,

        DATEDIFF('day',
            LAG(started_at) OVER (PARTITION BY profile_id ORDER BY started_at, game_id),
            started_at)                                                AS days_since_last_game,

        (ROW_NUMBER() OVER (PARTITION BY profile_id, civ
            ORDER BY started_at, game_id) - 1)                         AS civ_games_before,
        COALESCE(SUM(result) OVER (PARTITION BY profile_id, civ
            ORDER BY started_at, game_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0)      AS civ_wins_before
    FROM player_game
    WINDOW w_life AS (PARTITION BY profile_id ORDER BY started_at, game_id)
    """
    conn.execute(sql)


def build_onev1_snapshot(conn, mode: str, seasons: list[int]) -> bool:
    """
    Per (team game, player) 1v1-ladder skill AS OF just before the team match.

    Restricted to players who actually appear in the team games (keeps the window
    functions over the large 1v1 DB cheap). Returns False if the 1v1 DB is absent.
    """
    if not attach_1v1(conn):
        print("  1v1 database not found — skipping 1v1 cross-reference features.")
        conn.execute("CREATE OR REPLACE TABLE onev1_snap AS SELECT NULL::BIGINT AS game_id, "
                     "NULL::BIGINT AS profile_id WHERE FALSE")
        return False

    # Cumulative 1v1 history (inclusive of each row) for relevant players only.
    conn.execute(f"""
    CREATE OR REPLACE TABLE onev1_cum AS
    WITH relevant AS (
        SELECT DISTINCT p.profile_id
        FROM participants p
        JOIN games g ON p.game_id = g.game_id
        WHERE g.kind = '{mode}' AND g.season IN {_season_list_sql(seasons)}
    ),
    hist AS (
        SELECT
            op.profile_id,
            og.started_at,
            op.game_id,
            op.mmr   AS onev1_mmr,
            op.rating AS onev1_rating,
            op.result::INT AS result
        FROM onev1.participants op
        JOIN onev1.games og ON op.game_id = og.game_id
        JOIN relevant r ON r.profile_id = op.profile_id
        WHERE og.kind IN ('rm_1v1', 'rm_solo')
          AND og.started_at IS NOT NULL
          AND op.result IS NOT NULL
    )
    SELECT
        profile_id, started_at, game_id, onev1_mmr, onev1_rating,
        ROW_NUMBER() OVER w                                            AS onev1_games_incl,
        SUM(result) OVER (PARTITION BY profile_id ORDER BY started_at, game_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)          AS onev1_wins_incl,
        SUM(result) OVER (PARTITION BY profile_id ORDER BY started_at, game_id
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)                 AS onev1_wins_last20_incl,
        COUNT(*) OVER (PARTITION BY profile_id ORDER BY started_at, game_id
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)                 AS onev1_n_last20_incl
    FROM hist
    WINDOW w AS (PARTITION BY profile_id ORDER BY started_at, game_id)
    """)

    # ASOF: for each team participant, the most recent 1v1 row strictly before kickoff.
    conn.execute(f"""
    CREATE OR REPLACE TABLE onev1_snap AS
    SELECT
        tp.game_id,
        tp.profile_id,
        oc.onev1_mmr,
        oc.onev1_rating,
        oc.onev1_games_incl                 AS onev1_games,
        oc.onev1_wins_incl                  AS onev1_wins,
        oc.onev1_wins_last20_incl           AS onev1_recent_wins,
        oc.onev1_n_last20_incl              AS onev1_recent_n
    FROM (
        SELECT p.game_id, p.profile_id, g.started_at
        FROM participants p
        JOIN games g ON p.game_id = g.game_id
        WHERE g.kind = '{mode}' AND g.season IN {_season_list_sql(seasons)}
    ) tp
    ASOF LEFT JOIN onev1_cum oc
        ON tp.profile_id = oc.profile_id
       AND tp.started_at > oc.started_at
    """)
    return True


def build_team_agg(conn, mode: str, seasons: list[int]) -> None:
    """One row per (game_id, team_id) with all team-level features."""
    sql = f"""
    CREATE OR REPLACE TABLE team_agg AS
    WITH enriched AS (
        SELECT
            ps.game_id,
            ps.team_id,
            ps.profile_id,
            ps.result,
            ps.mmr,
            ps.skill,
            ps.civilization_randomized,
            ps.games_lifetime_before,
            ps.wins_lifetime_before,
            ps.wins_last10_before,
            ps.n_last10_before,
            ps.days_since_last_game,
            ps.civ_games_before,
            ps.civ_wins_before,
            os.onev1_mmr,
            os.onev1_games,
            os.onev1_wins,
            os.onev1_recent_wins,
            os.onev1_recent_n,
            -- team skill mean/std for the carry flag (window over the team)
            AVG(ps.skill)        OVER w  AS t_skill_mean,
            STDDEV_POP(ps.skill) OVER w  AS t_skill_std
        FROM player_stats ps
        LEFT JOIN onev1_snap os
               ON os.game_id = ps.game_id AND os.profile_id = ps.profile_id
        WINDOW w AS (PARTITION BY ps.game_id, ps.team_id)
    )
    SELECT
        game_id,
        team_id,
        MAX(result)                             AS team_result,
        MIN(profile_id)                         AS team_min_profile_id,
        COUNT(*)                                AS team_n,

        -- MMR aggregates (matchmaking balances on MMR)
        AVG(mmr)                                AS mmr_mean,
        MAX(mmr)                                AS mmr_max,
        MIN(mmr)                                AS mmr_min,
        COALESCE(STDDEV_POP(mmr), 0)            AS mmr_std,
        AVG(CASE WHEN mmr IS NULL THEN 1.0 ELSE 0.0 END) AS missing_mmr_frac,

        -- Skill aggregates (mmr w/ rating fallback) for dispersion / carry features
        AVG(skill)                              AS skill_mean,
        MAX(skill)                              AS skill_max,
        MIN(skill)                              AS skill_min,
        COALESCE(STDDEV_POP(skill), 0)          AS skill_std,

        -- ── Boost / carry exploit family ──────────────────────────────────────
        (MAX(skill) - MIN(skill))               AS carry_spread,
        -- carry gap: top player vs the mean of the *rest* of the team
        (MAX(skill) - (SUM(skill) - MAX(skill)) / NULLIF(COUNT(*) - 1, 0)) AS carry_gap,
        (MAX(skill) / NULLIF(AVG(skill), 0))    AS carry_ratio,
        SUM(CASE WHEN skill < {LOW_MMR_FLOOR} THEN 1 ELSE 0 END)          AS n_below_floor,
        SUM(CASE WHEN skill < t_skill_mean - {CARRY_STD_K} * t_skill_std
                 THEN 1 ELSE 0 END)             AS n_carried,

        -- History (team means of per-player leakage-free stats)
        AVG(games_lifetime_before)              AS games_lifetime_mean,
        AVG((wins_lifetime_before + {PRIOR_STRENGTH} * {GLOBAL_WR_PRIOR})
            / (games_lifetime_before + {PRIOR_STRENGTH}))                AS overall_wr_mean,
        AVG((wins_last10_before + {PRIOR_STRENGTH} * {GLOBAL_WR_PRIOR})
            / (n_last10_before + {PRIOR_STRENGTH}))                      AS recent_wr_mean,
        AVG(days_since_last_game)               AS days_since_mean,
        SUM(CASE WHEN games_lifetime_before < {NEW_PLAYER_THRESHOLD} THEN 1 ELSE 0 END)
                                                AS n_new_players,
        AVG((civ_wins_before + {PRIOR_STRENGTH} * {GLOBAL_WR_PRIOR})
            / (civ_games_before + {PRIOR_STRENGTH}))                     AS civ_wr_mean,
        SUM(CASE WHEN civilization_randomized THEN 1 ELSE 0 END)         AS n_random_civ,

        -- ── 1v1-ladder cross-reference ───────────────────────────────────────
        AVG(onev1_mmr)                          AS onev1_mmr_mean,
        MAX(onev1_mmr)                          AS onev1_mmr_max,
        AVG((onev1_wins + {PRIOR_STRENGTH} * {GLOBAL_WR_PRIOR})
            / (onev1_games + {PRIOR_STRENGTH}))                          AS onev1_wr_mean,
        AVG((onev1_recent_wins + {PRIOR_STRENGTH} * {GLOBAL_WR_PRIOR})
            / (onev1_recent_n + {PRIOR_STRENGTH}))                       AS onev1_recent_wr_mean,
        AVG(CASE WHEN onev1_mmr IS NOT NULL THEN 1.0 ELSE 0.0 END)       AS onev1_coverage,
        MAX(onev1_mmr - skill)                  AS onev1_max_minus_skill,
        SUM(CASE WHEN onev1_mmr - skill > {SMURF_GAP} THEN 1 ELSE 0 END) AS n_smurf_like
    FROM enriched
    GROUP BY game_id, team_id
    """
    conn.execute(sql)


# Columns carried per-team that become _a / _b in the match row.
_TEAM_COLS = [
    "mmr_mean", "mmr_max", "mmr_min", "mmr_std", "missing_mmr_frac",
    "skill_mean", "skill_max", "skill_min", "skill_std",
    "carry_spread", "carry_gap", "carry_ratio", "n_below_floor", "n_carried",
    "games_lifetime_mean", "overall_wr_mean", "recent_wr_mean", "days_since_mean",
    "n_new_players", "civ_wr_mean", "n_random_civ",
    "onev1_mmr_mean", "onev1_mmr_max", "onev1_wr_mean", "onev1_recent_wr_mean",
    "onev1_coverage", "onev1_max_minus_skill", "n_smurf_like",
]

# Of those, the directional ones whose A-vs-B diff is meaningful (and which flip
# sign under a team swap).
_DIFF_COLS = [
    "mmr_mean", "mmr_max", "mmr_min", "skill_mean", "skill_max", "skill_min",
    "carry_spread", "carry_gap", "carry_ratio", "n_below_floor", "n_carried",
    "games_lifetime_mean", "overall_wr_mean", "recent_wr_mean", "days_since_mean",
    "n_new_players", "civ_wr_mean",
    "onev1_mmr_mean", "onev1_mmr_max", "onev1_wr_mean", "onev1_recent_wr_mean",
    "onev1_max_minus_skill", "n_smurf_like",
]


def build_training_matrix(conn, mode: str, seasons: list[int], rebuild: bool = True,
                          with_premade: bool = True) -> pd.DataFrame:
    """Run the full pipeline and return one row per match (Team A = lower min profile_id)."""
    if rebuild or not table_exists(conn, "team_agg"):
        t0 = time.time()
        print("  Building player_stats ...", flush=True)
        build_player_stats(conn, mode, seasons)
        print("  Building 1v1 snapshot ...", flush=True)
        build_onev1_snapshot(conn, mode, seasons)
        print("  Building team_agg ...", flush=True)
        build_team_agg(conn, mode, seasons)
        print(f"  Feature tables built in {time.time()-t0:.1f}s", flush=True)

    team_cols = list(_TEAM_COLS)
    diff_cols = list(_DIFF_COLS)
    if with_premade:
        from .network import PREMADE_DIFF_COLS, PREMADE_TEAM_COLS, ensure_premade
        ensure_premade(conn)
        prem = ", ".join(f"pm.{c}" for c in PREMADE_TEAM_COLS)
        team = conn.execute(f"""
            SELECT ta.*, {prem}
            FROM team_agg ta
            LEFT JOIN team_premade_agg pm USING (game_id, team_id)
        """).df()
        for c in PREMADE_TEAM_COLS:
            team[c] = team[c].fillna(0)
        team_cols += PREMADE_TEAM_COLS
        diff_cols += PREMADE_DIFF_COLS
    else:
        team = conn.execute("SELECT * FROM team_agg").df()

    games = conn.execute(f"""
        SELECT game_id, started_at, duration, map, server, patch, season
        FROM games
        WHERE kind = '{mode}' AND season IN {_season_list_sql(seasons)}
    """).df()

    # Two rows per game → assign A (lower team_min_profile_id) and B.
    team = team.sort_values(["game_id", "team_min_profile_id"]).reset_index(drop=True)
    team["role"] = team.groupby("game_id").cumcount().map({0: "a", 1: "b"})

    a = team[team.role == "a"].set_index("game_id")
    b = team[team.role == "b"].set_index("game_id")
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]

    rows = pd.DataFrame(index=common)
    rows["target"] = a["team_result"].astype("Int64")
    for col in team_cols:
        rows[f"{col}_a"] = a[col].values
        rows[f"{col}_b"] = b[col].values
    for col in diff_cols:
        rows[f"{col}_diff"] = a[col].values - b[col].values

    if with_premade:
        # symmetric match-level premade flags (unchanged under a team swap)
        rows["both_premade"] = ((a["team_is_premade"].values > 0)
                                & (b["team_is_premade"].values > 0)).astype(int)
        rows["premade_xor"] = ((a["team_is_premade"].values > 0)
                               ^ (b["team_is_premade"].values > 0)).astype(int)

    out = rows.reset_index().merge(games, on="game_id", how="left")
    out = out[out["target"].notna()].copy()
    out["target"] = out["target"].astype(int)
    out["season"] = out["season"].astype("Int64").astype(str)
    return out


# Feature lists used by the model (built from the columns above).
NUMERIC_FEATURES = (
    [f"{c}_a" for c in _TEAM_COLS]
    + [f"{c}_b" for c in _TEAM_COLS]
    + [f"{c}_diff" for c in _DIFF_COLS]
)
CATEGORICAL_FEATURES = ["map", "server", "patch", "season"]
# Base feature set (no teammate-network / premade features) — used for the AUC-lift comparison.
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Premade (teammate-network) feature block, leakage-free via weekly snapshots.
from .network import PREMADE_DIFF_COLS as _PD, PREMADE_TEAM_COLS as _PT  # noqa: E402

PREMADE_NUMERIC = (
    [f"{c}_a" for c in _PT] + [f"{c}_b" for c in _PT]
    + [f"{c}_diff" for c in _PD] + ["both_premade", "premade_xor"]
)
ALL_FEATURES_PREMADE = NUMERIC_FEATURES + PREMADE_NUMERIC + CATEGORICAL_FEATURES

# The single feature for the matchmaking-quality baseline.
MMR_DIFF_FEATURE = "mmr_mean_diff"


def build_dataset(mode: str, seasons: list[int], db_path=None, rebuild: bool = True,
                  with_premade: bool = True) -> pd.DataFrame:
    conn = get_conn(db_path)
    try:
        return build_training_matrix(conn, mode, seasons, rebuild=rebuild,
                                     with_premade=with_premade)
    finally:
        conn.close()
