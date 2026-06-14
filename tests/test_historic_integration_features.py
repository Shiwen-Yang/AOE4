"""Tests for the historic-integration feature builders.

Verifies that:
  - build_recent_only_base_overrides caps base lifetime/civ counts to the last N
    prior games (honest aoe4world-page mock), and
  - build_career_block reports FULL-HISTORY career summaries computed leakage-free
    (current game's MMR excluded), surviving the recent-window cap in the dual model.
"""
from datetime import datetime, timedelta
from pathlib import Path
import sys

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoe4_predict.db import init_schema
from aoe4_predict.features import build_civ_matchup_priors, build_player_stats, build_training_features
from aoe4_predict.features_extra import (
    apply_api_cap_p1_p3_p4_p5_overrides,
    apply_career_block,
    apply_recent_only_base_overrides,
    build_api_cap_p1_p3_p4_p5_overrides,
    build_career_block,
    build_recent_only_base_overrides,
    extend_training_features,
)


def _insert_game(conn, game_id, started_at, duration, p1, p2):
    conn.execute(
        """
        INSERT INTO games (
            game_id, started_at, finished_at, duration, map_id, map, kind, server, patch, season, source_file
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [game_id, started_at, started_at + timedelta(seconds=duration), duration,
         1, "Dry Arabia", "rm_solo", "steam", "patch-test", 10, "synthetic"],
    )
    conn.executemany(
        """
        INSERT INTO participants (
            game_id, profile_id, result, civilization, civilization_randomized,
            rating, rating_diff, mmr, mmr_diff, input_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (game_id, p1[0], p1[1], p1[2], False, p1[3], 0, p1[4], 0, "mouse_keyboard"),
            (game_id, p2[0], p2[1], p2[2], False, p2[3], 0, p2[4], 0, "mouse_keyboard"),
        ],
    )


def _build_scenario(conn):
    """Player 1 plays 40 prior english games (mmr 1401..1440 vs unique opponents),
    then meets new player 2 in the target game 41."""
    init_schema(conn)
    base = datetime(2026, 1, 1, 0, 0, 0)
    # Player 1 wins the first 20 games, loses the rest. Full-history WR (0.5)
    # therefore differs from the last-30-games WR (10/30), so the cap is testable.
    for idx in range(1, 41):
        p1_win = idx <= 20
        _insert_game(
            conn, idx, base + timedelta(hours=idx - 1), 900,
            (1, p1_win, "english", 1400 + idx, 1400 + idx),
            (1000 + idx, not p1_win, "abbasid", 1200, 1200),
        )
    # Target meeting: current MMR 1600 must NOT leak into career peak/avg.
    _insert_game(
        conn, 41, base + timedelta(hours=40), 1200,
        (1, True, "english", 1600, 1600),
        (2, False, "french", 1700, 1700),
    )
    build_player_stats(conn)
    build_civ_matchup_priors(conn)
    build_training_features(conn, train_seasons=[10])
    full_df = extend_training_features(
        conn, None,
        {"civ_recency", "adjusted_form", "duration_profile", "head_to_head"},
    )
    return full_df


def test_recent_only_base_caps_lifetime_and_civ_counts():
    conn = duckdb.connect(":memory:")
    try:
        full_df = _build_scenario(conn)
        base_over = build_recent_only_base_overrides(conn, visible_match_cap=30)
        capped_df = apply_recent_only_base_overrides(full_df.copy(), base_over)

        full = full_df.loc[full_df["game_id"] == 41].iloc[0]
        capped = capped_df.loc[capped_df["game_id"] == 41].iloc[0]

        # Full history sees all 40 prior games; the cap sees only the last 30.
        assert full["games_lifetime_a"] == 40
        assert capped["games_lifetime_a"] == 30
        assert full["civ_games_a"] == 40
        assert capped["civ_games_a"] == 30
        # overall_wr must be re-derived from the capped counts (not stale).
        assert capped["overall_wr_a"] != pytest.approx(full["overall_wr_a"])
        # New player B has no prior games either way.
        assert capped["games_lifetime_b"] == 0
    finally:
        conn.close()


def test_career_block_is_full_history_and_leakage_free():
    conn = duckdb.connect(":memory:")
    try:
        full_df = _build_scenario(conn)
        career = build_career_block(conn)
        row = career.loc[career["game_id"] == 41].iloc[0]

        # Full-history career counts survive (independent of any cap).
        assert row["career_games_a"] == 40
        assert row["career_civ_games_a"] == 40
        # Leakage check: peak/avg use prior MMRs 1401..1440 only — never 1600.
        assert row["peak_mmr_a"] == 1440
        assert row["career_avg_mmr_a"] == pytest.approx(1420.5)
        # New player B: no prior games.
        assert row["career_games_b"] == 0
    finally:
        conn.close()


def test_dual_keeps_capped_base_and_full_history_career_together():
    conn = duckdb.connect(":memory:")
    try:
        full_df = _build_scenario(conn)
        api_over = build_api_cap_p1_p3_p4_p5_overrides(conn, visible_match_cap=30)
        base_over = build_recent_only_base_overrides(conn, visible_match_cap=30)
        career = build_career_block(conn)

        dual = apply_api_cap_p1_p3_p4_p5_overrides(full_df.copy(), api_over)
        dual = apply_recent_only_base_overrides(dual, base_over)
        dual = apply_career_block(dual, career)

        row = dual.loc[dual["game_id"] == 41].iloc[0]
        # Recent window: base lifetime capped to 30.
        assert row["games_lifetime_a"] == 30
        # Career block: full history retained as a separate column.
        assert row["career_games_a"] == 40
        # Derived career features present.
        assert "career_wr_a" in dual.columns
        assert "mmr_vs_peak_a" in dual.columns
        assert "form_vs_career_a" in dual.columns
        # mmr_vs_peak = current 1600 − peak 1440 = 160.
        assert row["mmr_vs_peak_a"] == pytest.approx(160.0)
    finally:
        conn.close()
