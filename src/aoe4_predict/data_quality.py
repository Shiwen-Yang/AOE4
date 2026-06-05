"""
Data quality report for the ingested DB.

Prints findings and saves them to QUALITY_REPORT_PATH as JSON.
Key checks:
  - Row counts
  - RM 1v1 game counts
  - Missingness by field and season
  - Civ coverage
  - MMR/rating availability by season (S3 has no MMR)
  - Season/patch distribution
"""
import json

from .config import QUALITY_REPORT_PATH
from .db import get_conn


def _pct(n, total):
    return f"{100 * n / max(total, 1):.1f}%"


def run_quality_report(conn=None, db_path=None, save: bool = True) -> dict:
    own_conn = conn is None
    if own_conn:
        conn = get_conn(db_path, read_only=True)

    report = {}

    # ── Row counts ──────────────────────────────────────────────────────────
    total_games = conn.execute("SELECT count(*) FROM games").fetchone()[0]
    total_parts = conn.execute("SELECT count(*) FROM participants").fetchone()[0]
    rm_games = conn.execute(
        "SELECT count(*) FROM games WHERE kind IN ('rm_1v1','rm_solo')"
    ).fetchone()[0]
    malformed = conn.execute(
        """
        SELECT count(*) FROM (
            SELECT game_id FROM participants GROUP BY game_id HAVING count(*) != 2
        )
        """
    ).fetchone()[0]

    report["row_counts"] = {
        "total_games": total_games,
        "rm_1v1_games": rm_games,
        "total_participants": total_parts,
        "malformed_games_not_exactly_2_participants": malformed,
    }

    print("\n=== Data Quality Report ===")
    print(f"Total games in DB:         {total_games:>10,}")
    print(f"RM 1v1 games:              {rm_games:>10,}")
    print(f"Total participant rows:    {total_parts:>10,}")
    print(f"Malformed (≠2 players):    {malformed:>10,}")

    # ── Missingness ─────────────────────────────────────────────────────────
    fields = ["result", "mmr", "rating", "civilization", "map", "patch", "started_at"]
    miss_rows = conn.execute(
        f"""
        SELECT
            {', '.join(f"sum(CASE WHEN p.{f} IS NULL THEN 1 ELSE 0 END)" for f in ['result','mmr','rating','civilization','input_type'])}
        FROM participants p
        JOIN games g ON p.game_id = g.game_id
        WHERE g.kind IN ('rm_1v1','rm_solo')
        """
    ).fetchone()
    miss_game = conn.execute(
        f"""
        SELECT
            {', '.join(f"sum(CASE WHEN {f} IS NULL THEN 1 ELSE 0 END)" for f in ['map','patch','started_at'])}
        FROM games
        WHERE kind IN ('rm_1v1','rm_solo')
        """
    ).fetchone()

    part_fields = ["result", "mmr", "rating", "civilization", "input_type"]
    game_fields = ["map", "patch", "started_at"]
    miss = {}
    print("\n── Participant-level missingness ──")
    for f, cnt in zip(part_fields, miss_rows):
        miss[f"missing_{f}"] = cnt
        print(f"  {f:<25} {cnt:>10,}  ({_pct(cnt, rm_games * 2)})")
    print("── Game-level missingness ──")
    for f, cnt in zip(game_fields, miss_game):
        miss[f"missing_game_{f}"] = cnt
        print(f"  {f:<25} {cnt:>10,}  ({_pct(cnt, rm_games)})")
    report["missingness"] = miss

    # ── MMR/rating availability by season ──────────────────────────────────
    season_miss = conn.execute(
        """
        SELECT
            g.season,
            count(*) AS participant_rows,
            sum(CASE WHEN p.mmr IS NULL THEN 1 ELSE 0 END) AS missing_mmr,
            sum(CASE WHEN p.rating IS NULL THEN 1 ELSE 0 END) AS missing_rating
        FROM participants p
        JOIN games g ON p.game_id = g.game_id
        WHERE g.kind IN ('rm_1v1','rm_solo')
        GROUP BY g.season
        ORDER BY g.season
        """
    ).df()

    print("\n── MMR / Rating missingness by season ──")
    print(season_miss.to_string(index=False))
    report["season_missingness"] = season_miss.to_dict(orient="records")

    # ── Games per season ────────────────────────────────────────────────────
    season_counts = conn.execute(
        """
        SELECT season, count(*) AS games
        FROM games WHERE kind IN ('rm_1v1','rm_solo')
        GROUP BY season ORDER BY season
        """
    ).df()
    print("\n── Games per season ──")
    print(season_counts.to_string(index=False))
    report["games_per_season"] = season_counts.to_dict(orient="records")

    # ── Unique civs ─────────────────────────────────────────────────────────
    civs = conn.execute(
        """
        SELECT civilization, count(*) AS appearances
        FROM participants p JOIN games g ON p.game_id = g.game_id
        WHERE g.kind IN ('rm_1v1','rm_solo') AND civilization IS NOT NULL
        GROUP BY civilization ORDER BY appearances DESC
        """
    ).df()
    civ_list = civs["civilization"].tolist()
    print(f"\n── Civilizations ({len(civ_list)} unique) ──")
    for row in civs.itertuples(index=False):
        print(f"  {row.civilization:<30} {row.appearances:>10,}")
    report["civilizations"] = civ_list
    report["civ_appearances"] = civs.to_dict(orient="records")

    # ── Games with no conclusive result ─────────────────────────────────────
    no_result = conn.execute(
        """
        SELECT count(*) FROM (
            SELECT game_id FROM participants
            GROUP BY game_id
            HAVING sum(CASE WHEN result IS NULL THEN 1 ELSE 0 END) > 0
        )
        """
    ).fetchone()[0]
    report["games_with_missing_result"] = no_result
    print(f"\n── Games with any missing result: {no_result:,}")

    # ── MMR reset check: compare first observed MMR in each season ──────────
    mmr_continuity = conn.execute(
        """
        WITH season_endpoints AS (
            SELECT
                p.profile_id,
                g.season,
                MIN_BY(p.mmr, g.started_at) AS mmr_season_start,
                MAX_BY(p.mmr, g.started_at) AS mmr_season_end
            FROM participants p
            JOIN games g ON p.game_id = g.game_id
            WHERE g.kind IN ('rm_1v1','rm_solo') AND p.mmr IS NOT NULL
            GROUP BY p.profile_id, g.season
        ),
        transitions AS (
            SELECT
                a.season                             AS season_a,
                b.season                             AS season_b,
                abs(b.mmr_season_start - a.mmr_season_end) AS mmr_jump
            FROM season_endpoints a
            JOIN season_endpoints b
                ON a.profile_id = b.profile_id
               AND b.season = a.season + 1
            WHERE a.mmr_season_end IS NOT NULL AND b.mmr_season_start IS NOT NULL
        )
        SELECT
            season_a,
            season_b,
            count(*)                                                 AS player_transitions,
            round(avg(mmr_jump), 1)                                  AS avg_mmr_jump,
            round(median(mmr_jump), 1)                               AS median_mmr_jump,
            round(percentile_cont(0.9) WITHIN GROUP (ORDER BY mmr_jump), 1) AS p90_mmr_jump
        FROM transitions
        GROUP BY season_a, season_b ORDER BY season_a
        LIMIT 20
        """
    ).df()

    if not mmr_continuity.empty:
        print("\n── MMR continuity across season boundaries ──")
        print(mmr_continuity.to_string(index=False))
        print("(Large median/p90 jumps suggest soft/hard resets between seasons)")
        report["mmr_continuity"] = mmr_continuity.to_dict(orient="records")
    else:
        report["mmr_continuity"] = []

    if save:
        QUALITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        QUALITY_REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nReport saved to {QUALITY_REPORT_PATH}")

    if own_conn:
        conn.close()

    return report
