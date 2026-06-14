"""Audit capped P1+P3+P4+P5 semantics against raw prior-game recomputation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aoe4_predict.config import GLOBAL_WR_PRIOR, PRIOR_STRENGTH
from aoe4_predict.db import get_conn
from aoe4_predict.features import build_civ_matchup_priors, build_player_stats, build_training_features
from aoe4_predict.features_extra import (
    apply_api_cap_p1_p3_p4_p5_overrides,
    build_api_cap_p1_p3_p4_p5_overrides,
    extend_training_features,
)

FAMILIES = {"civ_recency", "adjusted_form", "duration_profile", "head_to_head"}


def _smooth(wins: float, games: float) -> float:
    return float((wins + PRIOR_STRENGTH * GLOBAL_WR_PRIOR) / (games + PRIOR_STRENGTH))


def _fetch_prior_games(conn, profile_id: int, game_id: int, started_at) -> list[dict]:
    rows = conn.execute(
        """
        SELECT
            g.game_id,
            g.started_at,
            g.duration,
            p.civilization AS civ,
            p.result::INT AS result
        FROM participants p
        JOIN games g ON p.game_id = g.game_id
        WHERE p.profile_id = ?
          AND g.kind IN ('rm_1v1', 'rm_solo')
          AND p.result IS NOT NULL
          AND g.started_at IS NOT NULL
          AND (
                g.started_at < ?
                OR (g.started_at = ? AND g.game_id < ?)
          )
        ORDER BY g.started_at, g.game_id
        """,
        [profile_id, started_at, started_at, game_id],
    ).fetchall()
    return [
        {"game_id": r[0], "started_at": r[1], "duration": r[2], "civ": r[3], "result": int(r[4])}
        for r in rows
    ]


def _compute_side(prior_games: list[dict], civ: str, started_at, cap: int) -> dict[str, float | int | None]:
    visible = prior_games[-cap:]
    civ_games = [g for g in visible if g["civ"] == civ]

    def window(days: int) -> tuple[int, int]:
        cutoff = pd.Timestamp(started_at) - pd.Timedelta(days=days)
        items = [g for g in civ_games if pd.Timestamp(g["started_at"]) >= cutoff]
        return len(items), int(sum(g["result"] for g in items))

    g7, w7 = window(7)
    g30, w30 = window(30)
    g60, w60 = window(60)
    last_civ = civ_games[-1]["started_at"] if civ_games else None

    visible20 = visible[-20:]
    short_games = [g for g in visible if g["duration"] is not None and g["duration"] <= 900]
    long_games = [g for g in visible if g["duration"] is not None and g["duration"] > 1800]
    civ_dur = [g["duration"] for g in civ_games if g["duration"] is not None]
    dur20 = [g["duration"] for g in visible20 if g["duration"] is not None]
    durations = [g["duration"] for g in visible if g["duration"] is not None]

    recent = {}
    for n in (5, 10, 20):
        lastn = visible[-n:]
        wins = int(sum(g["result"] for g in lastn))
        recent[f"recent_w_{n}"] = wins
        recent[f"recent_wr_{n}"] = _smooth(wins, len(lastn))

    return {
        "visible_games": len(visible),
        "civ_games_7d": g7,
        "civ_wins_7d": w7,
        "civ_games_30d": g30,
        "civ_wins_30d": w30,
        "civ_games_60d": g60,
        "civ_wins_60d": w60,
        "days_since_civ": (
            (pd.Timestamp(started_at).date() - pd.Timestamp(last_civ).date()).days
            if last_civ is not None else None
        ),
        "act_games_30d": int(sum(pd.Timestamp(g["started_at"]) >= pd.Timestamp(started_at) - pd.Timedelta(days=30) for g in visible)),
        "act_games_60d": int(sum(pd.Timestamp(g["started_at"]) >= pd.Timestamp(started_at) - pd.Timedelta(days=60) for g in visible)),
        "avg_dur_life": float(np.mean(durations)) if durations else None,
        "civ_avg_dur": float(np.mean(civ_dur)) if civ_dur else None,
        "avg_dur_20": float(np.mean(dur20)) if dur20 else None,
        "short_games": len(short_games),
        "short_wins": int(sum(g["result"] for g in short_games)),
        "long_games": len(long_games),
        "long_wins": int(sum(g["result"] for g in long_games)),
        **recent,
    }, {g["game_id"] for g in visible}


def _compare_value(expected, actual, tol: float = 1e-6) -> bool:
    if expected is None and pd.isna(actual):
        return True
    if expected is None:
        return False
    if isinstance(expected, float):
        return abs(expected - float(actual)) <= tol
    return expected == actual


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit API-capped P1/P3/P4/P5 semantics on a real sample.")
    parser.add_argument("--db", default=None, help="Path to DuckDB file (default: aoe4.duckdb)")
    parser.add_argument("--seasons", default="10,11,12", help="Comma-separated seasons")
    parser.add_argument("--api-cap", type=int, default=50, help="Visible prior-game cap to audit")
    parser.add_argument("--sample-size", type=int, default=50, help="Number of training rows to audit")
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional JSON report path (default: reports/generated/api_cap_audit_<cap>.json)",
    )
    args = parser.parse_args()

    seasons = [int(s) for s in args.seasons.split(",") if s]
    report_path = Path(args.report_path) if args.report_path else ROOT / "reports" / "generated" / f"api_cap_audit_{args.api_cap}.json"
    db_path = Path(args.db) if args.db else None

    conn = get_conn(db_path)
    try:
        build_player_stats(conn)
        build_civ_matchup_priors(conn)
        df = build_training_features(conn, train_seasons=seasons)
        df = extend_training_features(conn, df, FAMILIES)
        overrides = build_api_cap_p1_p3_p4_p5_overrides(conn, visible_match_cap=args.api_cap)
    finally:
        conn.close()

    capped_df = apply_api_cap_p1_p3_p4_p5_overrides(df.copy(), overrides)
    candidates = df.loc[
        (df["games_lifetime_a"] >= args.api_cap) | (df["games_lifetime_b"] >= args.api_cap)
    ].copy()
    sample_df = candidates.sample(n=min(args.sample_size, len(candidates)), random_state=42)

    raw_override_mismatches: list[dict] = []
    p3_invariance_mismatches: list[dict] = []
    audited_rows: list[int] = []
    conn = get_conn(db_path, read_only=True)
    try:
        for _, row in sample_df.iterrows():
            game_id = int(row["game_id"])
            started_at = row["started_at"]
            audited_rows.append(game_id)

            prior_a = _fetch_prior_games(conn, int(row["profile_id_a"]), game_id, started_at)
            prior_b = _fetch_prior_games(conn, int(row["profile_id_b"]), game_id, started_at)
            side_a, visible_a_ids = _compute_side(prior_a, row["civ_a"], started_at, args.api_cap)
            side_b, visible_b_ids = _compute_side(prior_b, row["civ_b"], started_at, args.api_cap)
            intersection = visible_a_ids & visible_b_ids

            capped_row = capped_df.loc[capped_df["game_id"] == game_id].iloc[0]
            full_row = df.loc[df["game_id"] == game_id].iloc[0]

            checks = {
                "visible_games_a": side_a["visible_games"],
                "visible_games_b": side_b["visible_games"],
                "civ_games_60d_a": side_a["civ_games_60d"],
                "civ_wins_60d_a": side_a["civ_wins_60d"],
                "civ_games_60d_b": side_b["civ_games_60d"],
                "civ_wins_60d_b": side_b["civ_wins_60d"],
                "days_since_civ_a": side_a["days_since_civ"],
                "days_since_civ_b": side_b["days_since_civ"],
                "avg_dur_life_a": side_a["avg_dur_life"],
                "avg_dur_life_b": side_b["avg_dur_life"],
                "avg_dur_20_a": side_a["avg_dur_20"],
                "avg_dur_20_b": side_b["avg_dur_20"],
                "civ_avg_dur_a": side_a["civ_avg_dur"],
                "civ_avg_dur_b": side_b["civ_avg_dur"],
                "short_games_a": side_a["short_games"],
                "short_wins_a": side_a["short_wins"],
                "short_games_b": side_b["short_games"],
                "short_wins_b": side_b["short_wins"],
                "long_games_a": side_a["long_games"],
                "long_wins_a": side_a["long_wins"],
                "long_games_b": side_b["long_games"],
                "long_wins_b": side_b["long_wins"],
                "h2h_games": len(intersection),
                "h2h_wins_a": int(sum(g["result"] for g in prior_a if g["game_id"] in intersection)),
            }

            for field, expected in checks.items():
                actual = overrides.loc[overrides["game_id"] == game_id].iloc[0][field]
                if not _compare_value(expected, actual):
                    raw_override_mismatches.append({"game_id": game_id, "field": field, "expected": expected, "actual": None if pd.isna(actual) else actual})

            for side, expected_recent in (("a", side_a), ("b", side_b)):
                for n in (5, 10, 20):
                    for suffix in ("recent_w", "recent_wr"):
                        full_val = full_row[f"{suffix}_{n}_{side}"]
                        capped_val = capped_row[f"{suffix}_{n}_{side}"]
                        if not _compare_value(full_val, capped_val, tol=1e-5):
                            p3_invariance_mismatches.append({
                                "game_id": game_id,
                                "field": f"{suffix}_{n}_{side}",
                                "expected": None if pd.isna(full_val) else full_val,
                                "actual": None if pd.isna(capped_val) else capped_val,
                            })
    finally:
        conn.close()

    report = {
        "api_cap": args.api_cap,
        "sample_size_requested": args.sample_size,
        "sample_size_audited": len(audited_rows),
        "audited_game_ids": audited_rows,
        "raw_override_mismatch_count": len(raw_override_mismatches),
        "raw_override_mismatches": raw_override_mismatches[:200],
        "p3_invariance_mismatch_count": len(p3_invariance_mismatches),
        "p3_invariance_mismatches": p3_invariance_mismatches[:200],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({
        "report_path": str(report_path),
        "api_cap": args.api_cap,
        "sample_size_audited": len(audited_rows),
        "raw_override_mismatch_count": len(raw_override_mismatches),
        "p3_invariance_mismatch_count": len(p3_invariance_mismatches),
    }, indent=2))


if __name__ == "__main__":
    main()
