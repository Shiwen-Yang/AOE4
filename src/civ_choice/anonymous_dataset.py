"""DuckDB dataset builder for anonymous opponent civ-choice prediction."""
from __future__ import annotations

import time

import pandas as pd


_ANON_PLAYER_RAW_SQL = """
CREATE OR REPLACE TABLE anonymous_player_raw_games AS
SELECT
    p.game_id,
    p.profile_id,
    p.civilization AS civ,
    p.result::INT AS result,
    p.mmr AS player_mmr,
    p.rating AS player_rating,
    CASE
        WHEN p.mmr IS NULL THEN 'unknown'
        WHEN p.mmr < 1000 THEN 'low'
        WHEN p.mmr < 1400 THEN 'mid'
        ELSE 'high'
    END AS mmr_bucket,
    CASE
        WHEN p.rating IS NULL THEN 'unknown'
        WHEN p.rating < 1000 THEN 'low'
        WHEN p.rating < 1400 THEN 'mid'
        ELSE 'high'
    END AS rating_bucket,
    g.started_at,
    g.season,
    COALESCE(g.patch, 'unknown') AS patch,
    COALESCE(g.map, 'unknown') AS map
FROM participants p
JOIN games g ON p.game_id = g.game_id
WHERE g.kind IN ('rm_1v1', 'rm_solo')
  AND p.result IS NOT NULL
  AND p.civilization IS NOT NULL
  AND COALESCE(p.civilization_randomized, FALSE) = FALSE
  AND g.started_at IS NOT NULL
"""

_ANON_CIV_FIRST_SEEN_SQL = """
CREATE OR REPLACE TABLE anonymous_civ_first_seen AS
SELECT civilization AS civ, MIN(g.started_at) AS first_seen_at
FROM participants p
JOIN games g ON p.game_id = g.game_id
WHERE p.civilization IS NOT NULL
  AND COALESCE(p.civilization_randomized, FALSE) = FALSE
  AND g.started_at IS NOT NULL
GROUP BY civilization
"""

_ANON_TARGET_GAMES_SQL = """
CREATE OR REPLACE TABLE anonymous_opponent_player_games AS
SELECT
    target.game_id,
    target.profile_id,
    {user_profile_select} AS user_profile_id,
    target.civ AS chosen_civ,
    target.result,
    target.player_mmr,
    target.player_rating,
    target.mmr_bucket,
    target.rating_bucket,
    target.started_at,
    target.season,
    COALESCE(target.patch, 'unknown') AS patch,
    COALESCE(target.map, 'unknown') AS map
FROM anonymous_player_raw_games target
{user_join}
WHERE target.season IN ({seasons})
  {target_filter}
"""

_ANON_CANDIDATE_ROWS_SQL = """
CREATE OR REPLACE TABLE anonymous_opponent_candidate_rows AS
SELECT
    pg.game_id,
    pg.profile_id,
    pg.user_profile_id,
    pg.chosen_civ,
    pg.result,
    pg.player_mmr,
    pg.player_rating,
    pg.mmr_bucket,
    pg.rating_bucket,
    DATE_TRUNC('day', pg.started_at) AS started_day,
    pg.started_at,
    pg.season,
    pg.patch,
    pg.map,
    cfs.civ AS candidate_civ,
    (pg.chosen_civ = cfs.civ)::INT AS target
FROM anonymous_opponent_player_games pg
JOIN anonymous_civ_first_seen cfs
  ON cfs.first_seen_at <= pg.started_at
"""

_ANON_TRAINING_MATRIX_SQL = """
CREATE OR REPLACE TABLE anonymous_opponent_training_matrix AS
SELECT
    cr.game_id,
    cr.profile_id,
    cr.user_profile_id,
    cr.candidate_civ,
    cr.target,
    cr.chosen_civ,
    cr.started_at,
    cr.season,
    cr.patch,
    cr.map,
    cr.player_mmr,
    cr.player_rating,
    cr.mmr_bucket,
    cr.rating_bucket,

    COALESCE(global_hist.cand_count, 0) AS cand_global_prior_count,
    COALESCE(global_tot.total_count, 0) AS global_prior_total,
    COALESCE(mmr_hist.cand_count, 0) AS cand_mmr_bucket_count,
    COALESCE(mmr_tot.total_count, 0) AS mmr_bucket_total,
    COALESCE(rating_hist.cand_count, 0) AS cand_rating_bucket_count,
    COALESCE(rating_tot.total_count, 0) AS rating_bucket_total,
    COALESCE(map_hist.cand_count, 0) AS cand_map_count,
    COALESCE(map_tot.total_count, 0) AS map_total,
    COALESCE(map_patch_hist.cand_count, 0) AS cand_map_patch_count,
    COALESCE(map_patch_tot.total_count, 0) AS map_patch_total,
    COALESCE(patch_hist.cand_count, 0) AS cand_patch_count,
    COALESCE(patch_tot.total_count, 0) AS patch_total,

    {user_recent_select}
FROM anonymous_opponent_candidate_rows cr
LEFT JOIN anon_cum_global_civ global_hist
  ON global_hist.day = cr.started_day
 AND global_hist.civ = cr.candidate_civ
LEFT JOIN anon_cum_global_total global_tot
  ON global_tot.day = cr.started_day
LEFT JOIN anon_cum_mmr_bucket_civ mmr_hist
  ON mmr_hist.day = cr.started_day
 AND mmr_hist.mmr_bucket = cr.mmr_bucket
 AND mmr_hist.civ = cr.candidate_civ
LEFT JOIN anon_cum_mmr_bucket_total mmr_tot
  ON mmr_tot.day = cr.started_day
 AND mmr_tot.mmr_bucket = cr.mmr_bucket
LEFT JOIN anon_cum_rating_bucket_civ rating_hist
  ON rating_hist.day = cr.started_day
 AND rating_hist.rating_bucket = cr.rating_bucket
 AND rating_hist.civ = cr.candidate_civ
LEFT JOIN anon_cum_rating_bucket_total rating_tot
  ON rating_tot.day = cr.started_day
 AND rating_tot.rating_bucket = cr.rating_bucket
LEFT JOIN anon_cum_map_civ map_hist
  ON map_hist.day = cr.started_day
 AND map_hist.map = cr.map
 AND map_hist.civ = cr.candidate_civ
LEFT JOIN anon_cum_map_total map_tot
  ON map_tot.day = cr.started_day
 AND map_tot.map = cr.map
LEFT JOIN anon_cum_map_patch_civ map_patch_hist
  ON map_patch_hist.day = cr.started_day
 AND map_patch_hist.map = cr.map
 AND map_patch_hist.patch = cr.patch
 AND map_patch_hist.civ = cr.candidate_civ
LEFT JOIN anon_cum_map_patch_total map_patch_tot
  ON map_patch_tot.day = cr.started_day
 AND map_patch_tot.map = cr.map
 AND map_patch_tot.patch = cr.patch
LEFT JOIN anon_cum_patch_civ patch_hist
  ON patch_hist.day = cr.started_day
 AND patch_hist.patch = cr.patch
 AND patch_hist.civ = cr.candidate_civ
LEFT JOIN anon_cum_patch_total patch_tot
  ON patch_tot.day = cr.started_day
 AND patch_tot.patch = cr.patch
{user_recent_join}
"""

_USER_RECENT_SELECT_SQL = """
    COALESCE(user_recent.cand_count_10, 0) AS cand_user_recent_opp_count_10,
    COALESCE(user_recent.total_10, 0) AS user_recent_opp_games_10,
    COALESCE(user_recent.cand_count_30, 0) AS cand_user_recent_opp_count_30,
    COALESCE(user_recent.total_30, 0) AS user_recent_opp_games_30,
    COALESCE(user_recent.cand_count_50, 0) AS cand_user_recent_opp_count_50,
    COALESCE(user_recent.total_50, 0) AS user_recent_opp_games_50,
    COALESCE(user_recent.cand_same_map_30, 0) AS cand_user_recent_opp_same_map_count_30,
    COALESCE(user_recent.total_same_map_30, 0) AS user_recent_opp_same_map_games_30
"""

_GENERIC_USER_RECENT_SELECT_SQL = """
    0 AS cand_user_recent_opp_count_10,
    0 AS user_recent_opp_games_10,
    0 AS cand_user_recent_opp_count_30,
    0 AS user_recent_opp_games_30,
    0 AS cand_user_recent_opp_count_50,
    0 AS user_recent_opp_games_50,
    0 AS cand_user_recent_opp_same_map_count_30,
    0 AS user_recent_opp_same_map_games_30
"""

_USER_RECENT_JOIN_SQL = """
LEFT JOIN anon_user_recent_by_candidate user_recent
  ON user_recent.game_id = cr.game_id
 AND user_recent.profile_id = cr.profile_id
 AND user_recent.candidate_civ = cr.candidate_civ
"""


def _run(conn, sql: str, label: str) -> None:
    t0 = time.time()
    print(f"  {label}...", end="", flush=True)
    conn.execute(sql)
    print(f" done ({time.time() - t0:.1f}s)")


def _build_daily_cumulative_tables(conn) -> None:
    """Precompute as-of-start-of-day global counts for anonymous features."""
    _run(
        conn,
        """
        CREATE OR REPLACE TABLE anon_target_days AS
        SELECT DISTINCT started_day AS day
        FROM anonymous_opponent_candidate_rows
        """,
        "anon_target_days",
    )
    _run(
        conn,
        """
        CREATE OR REPLACE TABLE anon_civ_options AS
        SELECT DISTINCT candidate_civ AS civ
        FROM anonymous_opponent_candidate_rows
        """,
        "anon_civ_options",
    )
    _run(
        conn,
        """
        CREATE OR REPLACE TABLE anon_cum_global_civ AS
        WITH grid AS (
            SELECT d.day, c.civ
            FROM anon_target_days d
            CROSS JOIN anon_civ_options c
        ),
        daily AS (
            SELECT DATE_TRUNC('day', started_at) AS day, civ, COUNT(*) AS n
            FROM anonymous_player_raw_games
            GROUP BY 1, 2
        ),
        filled AS (
            SELECT g.day, g.civ, COALESCE(d.n, 0) AS n
            FROM grid g
            LEFT JOIN daily d ON d.day = g.day AND d.civ = g.civ
        )
        SELECT
            day,
            civ,
            COALESCE(
                SUM(n) OVER (
                    PARTITION BY civ
                    ORDER BY day
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ),
                0
            ) AS cand_count
        FROM filled
        """,
        "anon_cum_global_civ",
    )
    _run(
        conn,
        """
        CREATE OR REPLACE TABLE anon_cum_global_total AS
        WITH grid AS (
            SELECT day FROM anon_target_days
        ),
        daily AS (
            SELECT DATE_TRUNC('day', started_at) AS day, COUNT(*) AS n
            FROM anonymous_player_raw_games
            GROUP BY 1
        ),
        filled AS (
            SELECT g.day, COALESCE(d.n, 0) AS n
            FROM grid g
            LEFT JOIN daily d ON d.day = g.day
        )
        SELECT
            day,
            COALESCE(
                SUM(n) OVER (
                    ORDER BY day
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ),
                0
            ) AS total_count
        FROM filled
        """,
        "anon_cum_global_total",
    )

    def build_dim(prefix: str, cols: list[str]) -> None:
        col_list = ", ".join(cols)
        col_eq_daily = " AND ".join(f"d.{col} = g.{col}" for col in cols)
        col_eq_total = " AND ".join(f"d.{col} = g.{col}" for col in cols)
        partition_cols = ", ".join(cols + ["civ"])
        total_partition_cols = ", ".join(cols)
        grid_cols = ", ".join(f"v.{col}" for col in cols)
        select_grid_cols = ", ".join(f"g.{col}" for col in cols)
        select_cols = ", ".join(cols)

        _run(
            conn,
            f"""
            CREATE OR REPLACE TABLE anon_cum_{prefix}_civ AS
            WITH values AS (
                SELECT DISTINCT {col_list}
                FROM anonymous_opponent_candidate_rows
            ),
            grid AS (
                SELECT d.day, {grid_cols}, c.civ
                FROM anon_target_days d
                CROSS JOIN values v
                CROSS JOIN anon_civ_options c
            ),
            daily AS (
                SELECT
                    DATE_TRUNC('day', started_at) AS day,
                    {col_list},
                    civ,
                    COUNT(*) AS n
                FROM anonymous_player_raw_games
                GROUP BY 1, {", ".join(str(i) for i in range(2, len(cols) + 3))}
            ),
            filled AS (
                SELECT
                    g.day,
                    {select_grid_cols},
                    g.civ,
                    COALESCE(d.n, 0) AS n
                FROM grid g
                LEFT JOIN daily d
                  ON d.day = g.day
                 AND {col_eq_daily}
                 AND d.civ = g.civ
            )
            SELECT
                day,
                {select_cols},
                civ,
                COALESCE(
                    SUM(n) OVER (
                        PARTITION BY {partition_cols}
                        ORDER BY day
                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                    ),
                    0
                ) AS cand_count
            FROM filled
            """,
            f"anon_cum_{prefix}_civ",
        )
        _run(
            conn,
            f"""
            CREATE OR REPLACE TABLE anon_cum_{prefix}_total AS
            WITH values AS (
                SELECT DISTINCT {col_list}
                FROM anonymous_opponent_candidate_rows
            ),
            grid AS (
                SELECT d.day, {grid_cols}
                FROM anon_target_days d
                CROSS JOIN values v
            ),
            daily AS (
                SELECT
                    DATE_TRUNC('day', started_at) AS day,
                    {col_list},
                    COUNT(*) AS n
                FROM anonymous_player_raw_games
                GROUP BY 1, {", ".join(str(i) for i in range(2, len(cols) + 2))}
            ),
            filled AS (
                SELECT
                    g.day,
                    {select_grid_cols},
                    COALESCE(d.n, 0) AS n
                FROM grid g
                LEFT JOIN daily d
                  ON d.day = g.day
                 AND {col_eq_total}
            )
            SELECT
                day,
                {select_cols},
                COALESCE(
                    SUM(n) OVER (
                        PARTITION BY {total_partition_cols}
                        ORDER BY day
                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                    ),
                    0
                ) AS total_count
            FROM filled
            """,
            f"anon_cum_{prefix}_total",
        )

    build_dim("mmr_bucket", ["mmr_bucket"])
    build_dim("rating_bucket", ["rating_bucket"])
    build_dim("map", ["map"])
    build_dim("map_patch", ["map", "patch"])
    build_dim("patch", ["patch"])


def _build_user_recent_tables(conn, user_profile_id: int) -> None:
    """Precompute watched-user recent opponent civ counts by target candidate."""
    _run(
        conn,
        f"""
        CREATE OR REPLACE TABLE anon_user_opponent_games AS
        SELECT
            user_game.game_id,
            opp.profile_id AS opponent_profile_id,
            opp.civ AS opponent_civ,
            opp.map,
            user_game.started_at,
            ROW_NUMBER() OVER (
                ORDER BY user_game.started_at, user_game.game_id
            ) AS user_game_num
        FROM anonymous_player_raw_games user_game
        JOIN anonymous_player_raw_games opp
          ON opp.game_id = user_game.game_id
         AND opp.profile_id <> user_game.profile_id
        WHERE user_game.profile_id = {int(user_profile_id)}
        """,
        "anon_user_opponent_games",
    )
    _run(
        conn,
        """
        CREATE OR REPLACE TABLE anon_user_recent_by_candidate AS
        SELECT
            cr.game_id,
            cr.profile_id,
            cr.candidate_civ,
            COUNT(*) FILTER (
                WHERE hist.user_game_num >= target_user.user_game_num - 10
                  AND hist.opponent_civ = cr.candidate_civ
            ) AS cand_count_10,
            COUNT(*) FILTER (
                WHERE hist.user_game_num >= target_user.user_game_num - 10
            ) AS total_10,
            COUNT(*) FILTER (
                WHERE hist.user_game_num >= target_user.user_game_num - 30
                  AND hist.opponent_civ = cr.candidate_civ
            ) AS cand_count_30,
            COUNT(*) FILTER (
                WHERE hist.user_game_num >= target_user.user_game_num - 30
            ) AS total_30,
            COUNT(*) FILTER (
                WHERE hist.opponent_civ = cr.candidate_civ
            ) AS cand_count_50,
            COUNT(*) AS total_50,
            COUNT(*) FILTER (
                WHERE hist.user_game_num >= target_user.user_game_num - 30
                  AND hist.map = cr.map
                  AND hist.opponent_civ = cr.candidate_civ
            ) AS cand_same_map_30,
            COUNT(*) FILTER (
                WHERE hist.user_game_num >= target_user.user_game_num - 30
                  AND hist.map = cr.map
            ) AS total_same_map_30
        FROM anonymous_opponent_candidate_rows cr
        JOIN anon_user_opponent_games target_user
          ON target_user.game_id = cr.game_id
        LEFT JOIN anon_user_opponent_games hist
          ON hist.user_game_num < target_user.user_game_num
         AND hist.user_game_num >= target_user.user_game_num - 50
        GROUP BY cr.game_id, cr.profile_id, cr.candidate_civ
        """,
        "anon_user_recent_by_candidate",
    )


def _training_matrix_sql(*, include_user_recent: bool) -> str:
    return _ANON_TRAINING_MATRIX_SQL.format(
        user_recent_select=(
            _USER_RECENT_SELECT_SQL
            if include_user_recent
            else _GENERIC_USER_RECENT_SELECT_SQL
        ),
        user_recent_join=_USER_RECENT_JOIN_SQL if include_user_recent else "",
    )


def _build_anonymous_training_matrix(conn, *, include_user_recent: bool) -> None:
    _run(
        conn,
        _training_matrix_sql(include_user_recent=include_user_recent),
        "anonymous_opponent_training_matrix",
    )


def build_anonymous_tables(
    conn,
    seasons: list[int],
    *,
    user_profile_id: int | None = None,
    sample_mod: int = 10,
    sample_keep: int = 2,
) -> None:
    """Materialize anonymous-opponent civ-choice tables."""
    seasons_str = ", ".join(str(s) for s in seasons)
    conn.execute("SET memory_limit = '24GB'")
    conn.execute("SET temp_directory = '/tmp/duckdb_spill'")
    conn.execute("SET threads = 2")
    conn.execute("SET preserve_insertion_order = false")

    print("Building DuckDB tables for anonymous opponent civ-choice prediction...")
    _run(conn, _ANON_PLAYER_RAW_SQL, "anonymous_player_raw_games")
    _run(conn, _ANON_CIV_FIRST_SEEN_SQL, "anonymous_civ_first_seen")

    if user_profile_id is None:
        user_profile_select = "NULL::BIGINT"
        user_join = ""
        if sample_mod <= 0:
            raise ValueError("sample_mod must be positive")
        if sample_keep <= 0 or sample_keep > sample_mod:
            raise ValueError("sample_keep must be in [1, sample_mod]")
        target_filter = f"AND hash(target.game_id) % {sample_mod} < {sample_keep}"
    else:
        user_profile_select = f"{int(user_profile_id)}"
        user_join = (
            "JOIN anonymous_player_raw_games user_game\n"
            "  ON user_game.game_id = target.game_id\n"
            f" AND user_game.profile_id = {int(user_profile_id)}"
        )
        target_filter = f"AND target.profile_id <> {int(user_profile_id)}"

    _run(
        conn,
        _ANON_TARGET_GAMES_SQL.format(
            user_profile_select=user_profile_select,
            user_join=user_join,
            seasons=seasons_str,
            target_filter=target_filter,
        ),
        "anonymous_opponent_player_games",
    )
    _run(conn, _ANON_CANDIDATE_ROWS_SQL, "anonymous_opponent_candidate_rows")
    n_cand = conn.execute("SELECT COUNT(*) FROM anonymous_opponent_candidate_rows").fetchone()[0]
    n_groups = conn.execute(
        "SELECT COUNT(*) FROM anonymous_opponent_player_games"
    ).fetchone()[0]
    print(f"  Candidate rows: {n_cand:,} ({n_groups:,} player-matches × valid civs)")
    _build_daily_cumulative_tables(conn)
    if user_profile_id is not None:
        _build_user_recent_tables(conn, user_profile_id)
    _build_anonymous_training_matrix(conn, include_user_recent=user_profile_id is not None)


_STRING_COLS = [
    "candidate_civ",
    "chosen_civ",
    "map",
    "patch",
    "mmr_bucket",
    "rating_bucket",
]


def load_anonymous_training_matrix(conn) -> pd.DataFrame:
    print("  Loading anonymous training matrix into pandas...")
    df = conn.execute("SELECT * FROM anonymous_opponent_training_matrix").df()
    for col in _STRING_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def validate_anonymous_dataset(df: pd.DataFrame) -> dict:
    n_pos = int((df["target"] == 1).sum())
    n_total = len(df)
    n_groups = df.groupby(["game_id", "profile_id"]).ngroups
    n_civs_per_group = n_total / n_groups if n_groups else 0
    print("\n=== Anonymous Dataset Validation ===")
    print(f"  Total candidate rows:          {n_total:>10,}")
    print(f"  Player-match groups:           {n_groups:>10,}")
    print(f"  Avg candidate civs / group:    {n_civs_per_group:>10.1f}")
    print(f"  Target=1 rows (chosen civs):   {n_pos:>10,}")
    return {
        "n_total": n_total,
        "n_groups": n_groups,
        "n_civs_per_group": n_civs_per_group,
        "n_positive": n_pos,
    }
