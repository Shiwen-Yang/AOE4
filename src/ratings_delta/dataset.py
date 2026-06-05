"""Build participant-level rating delta dataset from the AOE4 DuckDB database."""
import sys

import pandas as pd

_DATASET_SQL = """
WITH
all_games AS (
    SELECT
        p.game_id,
        p.profile_id,
        p.result::INT   AS result,
        g.started_at,
        g.season
    FROM participants p
    JOIN games g ON p.game_id = g.game_id
    WHERE g.kind IN ('rm_1v1', 'rm_solo')
      AND p.result IS NOT NULL
      AND g.started_at IS NOT NULL
),

temporal_feats AS (
    SELECT
        game_id,
        profile_id,
        (ROW_NUMBER() OVER (
            PARTITION BY profile_id
            ORDER BY started_at, game_id
        ) - 1)                                                          AS games_lifetime_before,

        (ROW_NUMBER() OVER (
            PARTITION BY profile_id, season
            ORDER BY started_at, game_id
        ) - 1)                                                          AS games_season_before,

        DATEDIFF('day',
            LAG(started_at) OVER (
                PARTITION BY profile_id ORDER BY started_at, game_id
            ),
            started_at
        )                                                               AS days_since_last_game,

        COALESCE(SUM(result) OVER (
            PARTITION BY profile_id ORDER BY started_at, game_id
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
        ), 0) * 1.0 / NULLIF(
            LEAST(ROW_NUMBER() OVER (
                PARTITION BY profile_id ORDER BY started_at, game_id
            ) - 1, 10), 0
        )                                                               AS recent_wr_10,

        COALESCE(SUM(result) OVER (
            PARTITION BY profile_id ORDER BY started_at, game_id
            ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
        ), 0) * 1.0 / NULLIF(
            LEAST(ROW_NUMBER() OVER (
                PARTITION BY profile_id ORDER BY started_at, game_id
            ) - 1, 20), 0
        )                                                               AS recent_wr_20
    FROM all_games
),

streak_break AS (
    SELECT
        game_id, profile_id, started_at, result,
        CASE
            WHEN result = LAG(result) OVER (
                PARTITION BY profile_id ORDER BY started_at, game_id
            ) THEN 0 ELSE 1
        END AS break_flag
    FROM all_games
),

streak_group AS (
    SELECT *,
        SUM(break_flag) OVER (
            PARTITION BY profile_id
            ORDER BY started_at, game_id
            ROWS UNBOUNDED PRECEDING
        ) AS grp
    FROM streak_break
),

streak_running AS (
    SELECT
        profile_id, game_id, started_at,
        ROW_NUMBER() OVER (
            PARTITION BY profile_id, grp ORDER BY started_at, game_id
        )                                                               AS streak_len_incl,
        CASE WHEN result = 1 THEN 1 ELSE -1 END                       AS streak_sign
    FROM streak_group
),

streak_feats AS (
    SELECT
        profile_id, game_id,
        COALESCE(
            LAG(streak_sign * streak_len_incl::INT) OVER (
                PARTITION BY profile_id ORDER BY started_at, game_id
            ),
            0
        )                                                               AS current_streak
    FROM streak_running
),

main_rows AS (
    SELECT
        p.game_id,
        p.profile_id,
        p.result::INT                   AS result,
        p.rating                        AS player_rating_before,
        p.mmr                           AS player_mmr_before,
        p.rating_diff                   AS observed_rating_delta,
        p.mmr_diff                      AS observed_hidden_mmr_delta,
        g.started_at,
        g.season,
        g.patch,
        g.map
    FROM participants p
    JOIN games g ON p.game_id = g.game_id
    WHERE g.kind IN ('rm_1v1', 'rm_solo')
      AND p.result IS NOT NULL
      AND p.rating_diff IS NOT NULL
      AND g.season IN ({seasons})
),

opponent AS (
    SELECT
        p.game_id,
        p.profile_id                    AS opponent_profile_id,
        p.rating                        AS opponent_rating_before,
        p.mmr                           AS opponent_mmr_before
    FROM participants p
    JOIN games g ON p.game_id = g.game_id
    WHERE g.kind IN ('rm_1v1', 'rm_solo')
)

SELECT
    m.game_id,
    m.started_at,
    m.season,
    m.patch,
    m.map,
    m.profile_id,
    o.opponent_profile_id,
    m.player_rating_before,
    o.opponent_rating_before,
    (m.player_rating_before - o.opponent_rating_before)    AS visible_rating_gap,
    m.player_mmr_before,
    o.opponent_mmr_before,
    (m.player_mmr_before - o.opponent_mmr_before)         AS hidden_mmr_gap,
    m.result,
    m.observed_rating_delta,
    m.observed_hidden_mmr_delta,
    tf.games_lifetime_before,
    tf.games_season_before                                 AS games_this_season_before,
    tf.days_since_last_game,
    sf.current_streak,
    tf.recent_wr_10,
    tf.recent_wr_20,
    (m.player_rating_before IS NULL)::INT                  AS missing_player_rating,
    (o.opponent_rating_before IS NULL)::INT                AS missing_opponent_rating,
    (m.player_mmr_before IS NULL)::INT                     AS missing_player_mmr,
    (o.opponent_mmr_before IS NULL)::INT                   AS missing_opponent_mmr
FROM main_rows m
JOIN opponent o
    ON m.game_id = o.game_id
   AND m.profile_id != o.opponent_profile_id
JOIN temporal_feats tf
    ON tf.game_id = m.game_id
   AND tf.profile_id = m.profile_id
JOIN streak_feats sf
    ON sf.game_id = m.game_id
   AND sf.profile_id = m.profile_id
"""


def build_dataset(conn, seasons: list[int]) -> pd.DataFrame:
    """Build participant-level DataFrame for rating update analysis.

    Computes temporal features (streak, recent win rate, games before) over ALL
    seasons' history so that S10+ features are accurate even for early-career players.
    """
    seasons_str = ", ".join(str(s) for s in seasons)
    sql = _DATASET_SQL.format(seasons=seasons_str)
    print(f"  Building dataset for seasons {seasons} (may take 1–3 min)...")
    df = conn.execute(sql).df()
    print(f"  Loaded {len(df):,} participant rows.")
    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    """Print sanity-check table; raise if winner/loser sign is badly wrong."""
    total = len(df)
    n_rating = df["observed_rating_delta"].notna().sum()
    n_mmr = df["observed_hidden_mmr_delta"].notna().sum()

    winners = df[df["result"] == 1]
    losers = df[df["result"] == 0]

    w_pos = (winners["observed_rating_delta"] > 0).sum()
    w_nonpos = (winners["observed_rating_delta"] <= 0).sum()
    l_neg = (losers["observed_rating_delta"] < 0).sum()
    l_nonneg = (losers["observed_rating_delta"] >= 0).sum()

    w_pos_pct = w_pos / len(winners) * 100 if len(winners) else 0
    l_neg_pct = l_neg / len(losers) * 100 if len(losers) else 0

    pcts = df["observed_rating_delta"].quantile([0.05, 0.25, 0.50, 0.75, 0.95])

    stats = {
        "total_rows": total,
        "rating_delta_pct": n_rating / total * 100,
        "mmr_delta_pct": n_mmr / total * 100,
        "winner_positive_pct": w_pos_pct,
        "loser_negative_pct": l_neg_pct,
        "winner_nonpos_count": int(w_nonpos),
        "loser_nonneg_count": int(l_nonneg),
        "mean_delta_winner": winners["observed_rating_delta"].mean(),
        "mean_delta_loser": losers["observed_rating_delta"].mean(),
        "median_delta_winner": winners["observed_rating_delta"].median(),
        "median_delta_loser": losers["observed_rating_delta"].median(),
        "delta_min": df["observed_rating_delta"].min(),
        "delta_p5": pcts[0.05],
        "delta_p25": pcts[0.25],
        "delta_p50": pcts[0.50],
        "delta_p75": pcts[0.75],
        "delta_p95": pcts[0.95],
        "delta_max": df["observed_rating_delta"].max(),
    }

    print("\n=== Dataset Validation ===")
    print(f"  Total participant rows:              {stats['total_rows']:>10,}")
    print(f"  rating_diff present:                 {stats['rating_delta_pct']:>9.1f}%")
    print(f"  mmr_diff present (secondary):        {stats['mmr_delta_pct']:>9.1f}%")
    print(f"  Winners with positive delta:          {stats['winner_positive_pct']:>9.1f}%")
    print(f"  Losers with negative delta:           {stats['loser_negative_pct']:>9.1f}%")
    print(f"  Winners with non-positive delta:     {stats['winner_nonpos_count']:>10,}")
    print(f"  Losers with non-negative delta:      {stats['loser_nonneg_count']:>10,}")
    print(f"  Mean rating delta — winners:         {stats['mean_delta_winner']:>10.2f}")
    print(f"  Mean rating delta — losers:          {stats['mean_delta_loser']:>10.2f}")
    print(f"  Median rating delta — winners:       {stats['median_delta_winner']:>10.2f}")
    print(f"  Median rating delta — losers:        {stats['median_delta_loser']:>10.2f}")
    print(f"  Percentiles [p5, p25, p50, p75, p95]:")
    print(f"    [{stats['delta_p5']:.0f}, {stats['delta_p25']:.0f}, "
          f"{stats['delta_p50']:.0f}, {stats['delta_p75']:.0f}, {stats['delta_p95']:.0f}]")
    print(f"  Range: [{stats['delta_min']:.0f}, {stats['delta_max']:.0f}]")

    if w_pos_pct < 80:
        print(
            f"\nFATAL: Only {w_pos_pct:.1f}% of winners have a positive rating delta.",
            file=sys.stderr,
        )
        print(
            "rating_diff may not represent post-match gain. Check field semantics.",
            file=sys.stderr,
        )
        sys.exit(1)

    if w_pos_pct < 95:
        print(
            f"\nWARNING: {100 - w_pos_pct:.1f}% of winners have non-positive rating delta "
            f"({w_nonpos:,} rows). This may indicate ties, disconnects, or data noise."
        )

    return stats
