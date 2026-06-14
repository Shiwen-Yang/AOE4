"""
Build and materialize DuckDB tables for civ-choice prediction.

Pipeline (in order):
  1. player_raw_games            — all non-randomized RM 1v1 games (full history)
  2. player_civ_raw_games        — same, with per-civ cumulative stats
  3. civ_first_seen              — MIN(started_at) per civ (season-aware candidate set)
  4. civ_global_rates_by_season  — pick rate per (civ, season)
  5. civ_global_rates_by_patch   — pick rate per (civ, patch)
  6. civ_choice_player_games     — 20% game-level sample of target seasons
  7. civ_choice_candidate_rows   — sampled games × valid candidate civs
  8. civ_choice_training_matrix  — full feature table (LATERAL joins)

Leakage rule: all feature_time < match_started_at.
"""
import sys
import time

import pandas as pd

RANDOM_CIV = "random_civ"

# ── 1. Full player game history ───────────────────────────────────────────────
_PLAYER_RAW_GAMES_SQL = """
CREATE OR REPLACE TABLE player_raw_games AS
WITH base AS (
    SELECT
        p.game_id,
        p.profile_id,
        CASE
            WHEN p.civilization_randomized = TRUE THEN 'random_civ'
            ELSE p.civilization
        END                     AS civ,
        p.result::INT           AS result,
        p.mmr                   AS player_mmr,
        p.rating                AS player_rating,
        g.started_at,
        g.season,
        g.patch,
        g.map
    FROM participants p
    JOIN games g ON p.game_id = g.game_id
    WHERE g.kind IN ('rm_1v1', 'rm_solo')
      AND p.result IS NOT NULL
      AND p.civilization IS NOT NULL
      AND g.started_at IS NOT NULL
),
marked AS (
    SELECT
        *,
        CASE
            WHEN civ = LAG(civ) OVER (
                PARTITION BY profile_id
                ORDER BY started_at, game_id
            )
            THEN 0
            ELSE 1
        END AS run_start
    FROM base
),
runs AS (
    SELECT
        *,
        SUM(run_start) OVER (
            PARTITION BY profile_id
            ORDER BY started_at, game_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS civ_run_id
    FROM marked
),
positioned AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY profile_id, civ_run_id
            ORDER BY started_at, game_id
        ) AS current_civ_streak_len
    FROM runs
)
SELECT
    game_id,
    profile_id,
    civ,
    result,
    player_mmr,
    player_rating,
    started_at,
    season,
    patch,
    map,
    -- Civ played in the immediately preceding game (for candidate_is_last_civ)
    LAG(civ) OVER (
        PARTITION BY profile_id
        ORDER BY started_at, game_id
    ) AS prev_civ,
    COALESCE(
        LAG(current_civ_streak_len) OVER (
            PARTITION BY profile_id
            ORDER BY started_at, game_id
        ),
        0
    ) AS prev_civ_streak_len
FROM positioned
"""

# ── 2. Per-civ cumulative stats (for ASOF lifetime lookup) ───────────────────
_PLAYER_CIV_RAW_SQL = """
CREATE OR REPLACE TABLE player_civ_raw_games AS
SELECT
    game_id,
    profile_id,
    civ,
    result,
    started_at,
    season,
    patch,
    map,
    -- Cumulative game count including this game (used in ASOF lifetime lookup)
    ROW_NUMBER() OVER (
        PARTITION BY profile_id, civ
        ORDER BY started_at, game_id
    )                               AS civ_game_num,
    -- Cumulative wins including this game
    COALESCE(SUM(result) OVER (
        PARTITION BY profile_id, civ
        ORDER BY started_at, game_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ), 0)                           AS civ_wins_cumul
FROM player_raw_games
"""

# ── 3. First time each modeled civ option appeared in data ────────────────────
_CIV_FIRST_SEEN_SQL = """
CREATE OR REPLACE TABLE civ_first_seen AS
SELECT
    CASE
        WHEN civilization_randomized = TRUE THEN 'random_civ'
        ELSE civilization
    END             AS civ,
    MIN(started_at) AS first_seen_at
FROM participants p
JOIN games g ON p.game_id = g.game_id
WHERE civilization IS NOT NULL
GROUP BY 1
"""

# ── 4. Global civ pick rates by season ───────────────────────────────────────
_GLOBAL_RATES_SEASON_SQL = """
CREATE OR REPLACE TABLE civ_global_rates_by_season AS
WITH counts AS (
    SELECT season, civ, COUNT(*) AS n
    FROM player_raw_games
    GROUP BY season, civ
),
totals AS (
    SELECT season, SUM(n) AS total FROM counts GROUP BY season
)
SELECT c.season, c.civ, c.n::DOUBLE / t.total AS pick_rate
FROM counts c JOIN totals t ON c.season = t.season
"""

# ── 5. Global civ pick rates by patch ────────────────────────────────────────
_GLOBAL_RATES_PATCH_SQL = """
CREATE OR REPLACE TABLE civ_global_rates_by_patch AS
WITH counts AS (
    SELECT patch, civ, COUNT(*) AS n
    FROM player_raw_games
    GROUP BY patch, civ
),
totals AS (
    SELECT patch, SUM(n) AS total FROM counts GROUP BY patch
)
SELECT c.patch, c.civ, c.n::DOUBLE / t.total AS pick_rate
FROM counts c JOIN totals t ON c.patch = t.patch
"""

# ── 6. Sampled player games (target seasons, 20% of games) ───────────────────
_PLAYER_GAMES_SQL = """
CREATE OR REPLACE TABLE civ_choice_player_games AS
SELECT
    prg.game_id,
    prg.profile_id,
    prg.civ             AS chosen_civ,
    prg.result,
    prg.player_mmr,
    prg.player_rating,
    prg.prev_civ,
    prg.prev_civ_streak_len,
    prg.started_at,
    prg.season,
    prg.patch,
    prg.map
FROM player_raw_games prg
WHERE prg.season IN ({seasons})
  AND hash(prg.game_id) % 10 < 2
"""

# ── 7. Candidate rows = sampled games × valid civs at game time ───────────────
_CANDIDATE_ROWS_SQL = """
CREATE OR REPLACE TABLE civ_choice_candidate_rows AS
SELECT
    pg.game_id,
    pg.profile_id,
    pg.chosen_civ,
    pg.result,
    pg.player_mmr,
    pg.player_rating,
    pg.prev_civ,
    pg.prev_civ_streak_len,
    pg.started_at,
    pg.season,
    pg.patch,
    pg.map,
    cfs.civ             AS candidate_civ,
    (pg.chosen_civ = cfs.civ)::INT AS target
FROM civ_choice_player_games pg
JOIN civ_first_seen cfs
    ON cfs.first_seen_at <= pg.started_at
"""

# ── 8. Full training matrix (LATERAL joins for all stats) ────────────────────
# N = 18 fallback for global pick rate when civ is new/unseen in prior period
_TRAINING_MATRIX_SQL = """
CREATE OR REPLACE TABLE civ_choice_training_matrix AS
SELECT
    cr.game_id,
    cr.profile_id,
    cr.candidate_civ,
    cr.target,
    cr.chosen_civ,
    cr.prev_civ,
    cr.started_at,
    cr.season,
    cr.patch,
    cr.map,
    cr.player_mmr,
    cr.player_rating,
    cr.result,
    COALESCE(cr.prev_civ_streak_len, 0) AS player_current_streak_len,
    CASE WHEN cr.candidate_civ = cr.prev_civ
         THEN COALESCE(cr.prev_civ_streak_len, 0)
         ELSE 0 END                  AS candidate_current_streak_len,

    -- ── Lifetime cumulative stats for this candidate civ ─────────────────
    COALESCE(cum.civ_game_num, 0)       AS cand_games_lifetime,
    COALESCE(cum.civ_wins_cumul, 0)     AS cand_wins_lifetime,
    CASE WHEN cum.last_played_at IS NOT NULL
         THEN DATEDIFF('day', cum.last_played_at, cr.started_at)
         ELSE NULL END                  AS days_since_cand_civ,

    -- ── 30-day stats ─────────────────────────────────────────────────────
    COALESCE(rec30.games_30d, 0)        AS cand_games_30d,
    COALESCE(rec30.wins_30d, 0)         AS cand_wins_30d,
    COALESCE(recseq.games_last_1_games, 0) AS cand_games_last_1_games,
    COALESCE(recseq.games_last_2_games, 0) AS cand_games_last_2_games,
    COALESCE(recseq.games_last_3_games, 0) AS cand_games_last_3_games,
    COALESCE(recseq.games_last_5_games, 0) AS cand_games_last_5_games,
    COALESCE(recseq.games_last_10_games, 0) AS cand_games_last_10_games,
    COALESCE(rec20g.games_last_20_games, 0) AS cand_games_last_20_games,
    COALESCE(recseq.candidate_last_played_position, 21) AS candidate_last_played_position,
    COALESCE(recseq.switch_count_last_10_games, 0) AS recent_civ_switch_count_last_10_games,
    COALESCE(recseq.unique_civs_last_10_games, 0) AS recent_unique_civs_last_10_games,
    COALESCE(recseq.entropy_last_10_games, 0) AS recent_entropy_last_10_games,
    COALESCE(recseq.games_last_20_same_map, 0) AS cand_games_last_20_same_map,
    COALESCE(recseq.player_games_last_20_same_map, 0) AS player_games_last_20_same_map,

    -- ── Patch stats ──────────────────────────────────────────────────────
    COALESCE(recpatch.games_patch, 0)   AS cand_games_this_patch,
    COALESCE(recpatch.wins_patch, 0)    AS cand_wins_this_patch,

    -- ── Map stats ────────────────────────────────────────────────────────
    COALESCE(recmap.games_map, 0)       AS cand_games_this_map,
    COALESCE(recmap.wins_map, 0)        AS cand_wins_this_map,

    -- ── Player overall stats (all civs, before this game) ────────────────
    COALESCE(pov.games_lifetime, 0)     AS player_games_lifetime,
    COALESCE(pov.games_30d, 0)          AS player_games_30d,
    COALESCE(pov.games_this_patch, 0)   AS player_games_this_patch,
    COALESCE(pov.games_this_map, 0)     AS player_games_this_map,

    -- ── Global pick rates ────────────────────────────────────────────────
    COALESCE(gpr_ps.pick_rate, 1.0/19)  AS cand_global_pr_prev_season,
    COALESCE(gpr_pp.pick_rate, 1.0/19)  AS cand_global_pr_prev_patch,
    COALESCE(gpr_pmb.pick_rate, 1.0/19) AS cand_global_pr_patch_mmr_bucket,
    COALESCE(gpr_mp.pick_rate, 1.0/19)  AS cand_global_pr_map_patch,
    COALESCE(gpr_cs.pick_rate, 1.0/19)  AS cand_global_pr_this_season

FROM civ_choice_candidate_rows cr

-- ── Lifetime stats: most recent civ play strictly before this game ────────
LEFT JOIN LATERAL (
    SELECT
        pcr.civ_game_num,
        pcr.civ_wins_cumul,
        pcr.started_at AS last_played_at
    FROM player_civ_raw_games pcr
    WHERE pcr.profile_id = cr.profile_id
      AND pcr.civ = cr.candidate_civ
      AND pcr.started_at < cr.started_at
    ORDER BY pcr.started_at DESC, pcr.game_id DESC
    LIMIT 1
) cum ON TRUE

-- ── 30d aggregation ──────────────────────────────────────────────────────
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)                    AS games_30d,
        COALESCE(SUM(pcr.result), 0) AS wins_30d
    FROM player_civ_raw_games pcr
    WHERE pcr.profile_id = cr.profile_id
      AND pcr.civ = cr.candidate_civ
      AND pcr.started_at < cr.started_at
      AND pcr.started_at >= cr.started_at - INTERVAL '30 days'
) rec30 ON TRUE

-- ── Exact last-20-games aggregation ──────────────────────────────────────
LEFT JOIN LATERAL (
    SELECT COUNT(*) AS games_last_20_games
    FROM (
        SELECT prg.civ
        FROM player_raw_games prg
        WHERE prg.profile_id = cr.profile_id
          AND prg.started_at < cr.started_at
        ORDER BY prg.started_at DESC, prg.game_id DESC
        LIMIT 20
    ) recent_games
    WHERE recent_games.civ = cr.candidate_civ
) rec20g ON TRUE

-- ── Recent sequence features ────────────────────────────────────────────
LEFT JOIN LATERAL (
    WITH recent AS (
        SELECT
            prg.civ,
            prg.map,
            ROW_NUMBER() OVER (ORDER BY prg.started_at DESC, prg.game_id DESC) AS rn
        FROM player_raw_games prg
        WHERE prg.profile_id = cr.profile_id
          AND prg.started_at < cr.started_at
        ORDER BY prg.started_at DESC, prg.game_id DESC
        LIMIT 20
    ),
    recent_with_prev AS (
        SELECT
            rn,
            civ,
            map,
            LAG(civ) OVER (ORDER BY rn) AS prev_recent_civ
        FROM recent
    ),
    counts10 AS (
        SELECT civ, COUNT(*) AS n
        FROM recent
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
    )
    SELECT
        COUNT(*) FILTER (WHERE rn <= 1 AND civ = cr.candidate_civ) AS games_last_1_games,
        COUNT(*) FILTER (WHERE rn <= 2 AND civ = cr.candidate_civ) AS games_last_2_games,
        COUNT(*) FILTER (WHERE rn <= 3 AND civ = cr.candidate_civ) AS games_last_3_games,
        COUNT(*) FILTER (WHERE rn <= 5 AND civ = cr.candidate_civ) AS games_last_5_games,
        COUNT(*) FILTER (WHERE rn <= 10 AND civ = cr.candidate_civ) AS games_last_10_games,
        COALESCE(MIN(rn) FILTER (WHERE civ = cr.candidate_civ), 21) AS candidate_last_played_position,
        COUNT(*) FILTER (
            WHERE rn <= 10
              AND prev_recent_civ IS NOT NULL
              AND civ <> prev_recent_civ
        ) AS switch_count_last_10_games,
        COUNT(DISTINCT civ) FILTER (WHERE rn <= 10) AS unique_civs_last_10_games,
        (SELECT entropy FROM entropy10) AS entropy_last_10_games,
        COUNT(*) FILTER (
            WHERE rn <= 20
              AND map = cr.map
              AND civ = cr.candidate_civ
        ) AS games_last_20_same_map,
        COUNT(*) FILTER (
            WHERE rn <= 20
              AND map = cr.map
        ) AS player_games_last_20_same_map
    FROM recent_with_prev
) recseq ON TRUE

-- ── Patch aggregation ────────────────────────────────────────────────────
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)                    AS games_patch,
        COALESCE(SUM(pcr.result), 0) AS wins_patch
    FROM player_civ_raw_games pcr
    WHERE pcr.profile_id = cr.profile_id
      AND pcr.civ = cr.candidate_civ
      AND pcr.patch = cr.patch
      AND pcr.started_at < cr.started_at
) recpatch ON TRUE

-- ── Map aggregation ──────────────────────────────────────────────────────
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)                    AS games_map,
        COALESCE(SUM(pcr.result), 0) AS wins_map
    FROM player_civ_raw_games pcr
    WHERE pcr.profile_id = cr.profile_id
      AND pcr.civ = cr.candidate_civ
      AND pcr.map = cr.map
      AND pcr.started_at < cr.started_at
) recmap ON TRUE

-- ── Player overall game counts (all civs) ────────────────────────────────
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)                                                        AS games_lifetime,
        COUNT(*) FILTER (
            WHERE prg.started_at >= cr.started_at - INTERVAL '30 days'
        )                                                               AS games_30d,
        COUNT(*) FILTER (WHERE prg.patch = cr.patch)                   AS games_this_patch,
        COUNT(*) FILTER (WHERE prg.map = cr.map)                       AS games_this_map
    FROM player_raw_games prg
    WHERE prg.profile_id = cr.profile_id
      AND prg.started_at < cr.started_at
) pov ON TRUE

-- ── Global pick rate: previous season ────────────────────────────────────
LEFT JOIN civ_global_rates_by_season gpr_ps
    ON gpr_ps.civ = cr.candidate_civ
   AND gpr_ps.season = cr.season - 1

-- ── Global pick rate: previous patch (approximate via same-season prior patch) ──
LEFT JOIN LATERAL (
    SELECT pick_rate
    FROM civ_global_rates_by_patch gpr
    WHERE gpr.civ = cr.candidate_civ
      AND gpr.patch < cr.patch
    ORDER BY gpr.patch DESC
    LIMIT 1
) gpr_pp ON TRUE

-- ── Historical patch × MMR-bucket pick rate before this match ────────────
LEFT JOIN LATERAL (
    WITH hist AS (
        SELECT prg.civ
        FROM player_raw_games prg
        WHERE prg.patch = cr.patch
          AND prg.started_at < cr.started_at
          AND (
              CASE
                  WHEN prg.player_mmr IS NULL THEN 'unknown'
                  WHEN prg.player_mmr < 1000 THEN 'low'
                  WHEN prg.player_mmr < 1400 THEN 'mid'
                  ELSE 'high'
              END
          ) = (
              CASE
                  WHEN cr.player_mmr IS NULL THEN 'unknown'
                  WHEN cr.player_mmr < 1000 THEN 'low'
                  WHEN cr.player_mmr < 1400 THEN 'mid'
                  ELSE 'high'
              END
          )
    )
    SELECT
        COUNT(*) FILTER (WHERE civ = cr.candidate_civ)::DOUBLE
        / NULLIF(COUNT(*), 0) AS pick_rate
    FROM hist
) gpr_pmb ON TRUE

-- ── Historical map × patch pick rate before this match ───────────────────
LEFT JOIN LATERAL (
    SELECT
        COUNT(*) FILTER (WHERE prg.civ = cr.candidate_civ)::DOUBLE
        / NULLIF(COUNT(*), 0) AS pick_rate
    FROM player_raw_games prg
    WHERE prg.map = cr.map
      AND prg.patch = cr.patch
      AND prg.started_at < cr.started_at
) gpr_mp ON TRUE

-- ── Global pick rate: this season (from prior games — approximate with season total) ──
LEFT JOIN civ_global_rates_by_season gpr_cs
    ON gpr_cs.civ = cr.candidate_civ
   AND gpr_cs.season = cr.season
"""


def _run(conn, sql: str, label: str) -> None:
    t0 = time.time()
    print(f"  {label}...", end="", flush=True)
    conn.execute(sql)
    print(f" done ({time.time() - t0:.1f}s)")


def _build_training_matrix_chunked(conn, n_chunks: int = 4) -> None:
    """Build civ_choice_training_matrix in hash-partitioned chunks to bound peak memory.

    Each chunk filters candidate_rows to ~1/n_chunks of games via hash(game_id) % n_chunks,
    then runs the full LATERAL-join SQL on that slice.  Chunks 1+ use INSERT INTO so the
    final table is complete.
    """
    t_total = time.time()
    for chunk in range(n_chunks):
        header = (
            "CREATE OR REPLACE TABLE civ_choice_training_matrix AS"
            if chunk == 0
            else "INSERT INTO civ_choice_training_matrix"
        )
        sql = _TRAINING_MATRIX_SQL.replace(
            "CREATE OR REPLACE TABLE civ_choice_training_matrix AS",
            header,
        ).replace(
            "FROM civ_choice_candidate_rows cr",
            f"FROM (SELECT * FROM civ_choice_candidate_rows"
            f" WHERE hash(game_id) % {n_chunks} = {chunk}) cr",
        )
        t0 = time.time()
        print(
            f"  civ_choice_training_matrix chunk {chunk + 1}/{n_chunks}...",
            end="",
            flush=True,
        )
        conn.execute(sql)
        conn.execute("CHECKPOINT")  # flush written pages to disk, free buffer pool
        print(f" done ({time.time() - t0:.1f}s)")
    print(f"  training_matrix total: {time.time() - t_total:.1f}s")


def build_tables(conn, seasons: list[int]) -> None:
    """Materialize all DuckDB intermediate and training tables."""
    seasons_str = ", ".join(str(s) for s in seasons)

    # The LATERAL-join step uses ~5 GB per chunk (4 chunks × ~4.5M rows).
    # Reduce threads to 2 so DuckDB holds less parallel working state.
    conn.execute("SET memory_limit = '24GB'")
    conn.execute("SET temp_directory = '/tmp/duckdb_spill'")
    conn.execute("SET threads = 2")
    conn.execute("SET preserve_insertion_order = false")

    print("Building DuckDB tables for civ-choice prediction...")

    # Check randomized-civ percentage (informational)
    total = conn.execute(
        "SELECT COUNT(*) FROM participants p JOIN games g ON p.game_id = g.game_id"
        " WHERE g.kind IN ('rm_1v1','rm_solo') AND p.result IS NOT NULL"
        f" AND g.season IN ({seasons_str})"
    ).fetchone()[0]

    randomized = conn.execute(
        "SELECT COUNT(*) FROM participants p JOIN games g ON p.game_id = g.game_id"
        " WHERE g.kind IN ('rm_1v1','rm_solo') AND p.result IS NOT NULL"
        f" AND g.season IN ({seasons_str})"
        " AND (p.civilization IS NULL OR p.civilization_randomized = TRUE)"
    ).fetchone()[0]

    pct = randomized / total * 100 if total else 0
    print(f"  Randomized/null civ rows excluded: {randomized:,} / {total:,} ({pct:.1f}%)")

    _run(conn, _PLAYER_RAW_GAMES_SQL, "player_raw_games")
    _run(conn, _PLAYER_CIV_RAW_SQL, "player_civ_raw_games")
    _run(conn, _CIV_FIRST_SEEN_SQL, "civ_first_seen")
    _run(conn, _GLOBAL_RATES_SEASON_SQL, "civ_global_rates_by_season")
    _run(conn, _GLOBAL_RATES_PATCH_SQL, "civ_global_rates_by_patch")
    _run(conn, _PLAYER_GAMES_SQL.format(seasons=seasons_str), "civ_choice_player_games")
    _run(conn, _CANDIDATE_ROWS_SQL, "civ_choice_candidate_rows")

    # Report candidate row count before expensive matrix build
    n_cand = conn.execute("SELECT COUNT(*) FROM civ_choice_candidate_rows").fetchone()[0]
    n_games = conn.execute(
        "SELECT COUNT(DISTINCT game_id) FROM civ_choice_player_games"
    ).fetchone()[0]
    print(f"  Candidate rows: {n_cand:,} ({n_games:,} sampled games × valid civs)")

    _build_training_matrix_chunked(conn, n_chunks=4)

    n_matrix = conn.execute("SELECT COUNT(*) FROM civ_choice_training_matrix").fetchone()[0]
    print(f"  Training matrix: {n_matrix:,} rows")


_STRING_COLS = ["candidate_civ", "chosen_civ", "prev_civ", "map", "patch"]


def load_training_matrix(conn, pioneer: bool = True) -> pd.DataFrame:
    """Pull training matrix from DuckDB into pandas.

    pioneer=True (default): load only hash(game_id) % 4 < 3, i.e. ~15% of all
    games (3/4 of the 20% sample).  Set False to load the full 20% sample.
    String columns are immediately cast to category to avoid ~3 GB of Python
    string-object overhead.
    """
    print("  Loading training matrix into pandas...")
    where = "WHERE hash(game_id) % 4 < 3" if pioneer else ""
    df = conn.execute(
        f"SELECT * FROM civ_choice_training_matrix {where}"
    ).df()
    for col in _STRING_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns  (pioneer={pioneer})")
    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    """Print dataset sanity checks."""
    n_pos = (df["target"] == 1).sum()
    n_total = len(df)
    n_groups = df.groupby(["game_id", "profile_id"]).ngroups
    n_civs_per_group = n_total / n_groups if n_groups else 0

    print("\n=== Dataset Validation ===")
    print(f"  Total candidate rows:          {n_total:>10,}")
    print(f"  Player-match groups:           {n_groups:>10,}")
    print(f"  Avg candidate civs / group:    {n_civs_per_group:>10.1f}")
    print(f"  Target=1 rows (chosen civs):   {n_pos:>10,}")
    print(f"  Seasons: {sorted(df['season'].unique().tolist())}")
    print(f"  Candidate civs: {sorted(df['candidate_civ'].unique().tolist())}")

    civ_dist = df[df["target"] == 1]["candidate_civ"].value_counts(normalize=True).head(5)
    print("\n  Top 5 chosen civs (target=1):")
    for civ, share in civ_dist.items():
        print(f"    {civ:<30}  {share:.3f}")

    return {
        "n_total": n_total,
        "n_groups": n_groups,
        "n_civs_per_group": n_civs_per_group,
        "n_positive": n_pos,
    }
