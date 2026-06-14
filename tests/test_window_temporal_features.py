"""Tests for the window-temporal block (aoe4world-page-only features).

Verifies the span / density / gap / activity descriptors are computed over the
last-N visible window using only timestamps available on the recent-games page,
and exclude the current match (leakage-safe).
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
    apply_window_temporal_overrides,
    build_window_temporal_overrides,
    extend_training_features,
)


def _insert_game(conn, game_id, started_at, p1, p2, duration=900):
    conn.execute(
        """
        INSERT INTO games (
            game_id, started_at, finished_at, duration, map_id, map, kind, server, patch, season, source_file
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [game_id, started_at, started_at + timedelta(seconds=duration), duration,
         1, "Dry Arabia", "rm_solo", "steam", "p", 10, "syn"],
    )
    conn.executemany(
        """
        INSERT INTO participants (
            game_id, profile_id, result, civilization, civilization_randomized,
            rating, rating_diff, mmr, mmr_diff, input_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (game_id, p1[0], p1[1], p1[2], False, p1[3], 0, p1[4], 0, "mk"),
            (game_id, p2[0], p2[1], p2[2], False, p2[3], 0, p2[4], 0, "mk"),
        ],
    )


def test_window_temporal_span_density_gaps_and_leakage():
    conn = duckdb.connect(":memory:")
    try:
        init_schema(conn)
        base = datetime(2026, 1, 1, 0, 0, 0)
        # Player 1: 5 prior games on days 0, 1, 2, 12, 13 (a 10-day break before
        # the 4th), then the target match on day 20.
        prior_days = [0, 1, 2, 12, 13]
        for idx, d in enumerate(prior_days, start=1):
            _insert_game(
                conn, idx, base + timedelta(days=d),
                (1, idx % 2 == 0, "english", 1400, 1400),
                (1000 + idx, idx % 2 != 0, "french", 1300, 1300),
            )
        _insert_game(
            conn, 99, base + timedelta(days=20),
            (1, True, "english", 1500, 1500),
            (2, False, "french", 1600, 1600),
        )

        build_player_stats(conn)
        build_civ_matchup_priors(conn)
        build_training_features(conn, train_seasons=[10])
        full = extend_training_features(
            conn, None,
            {"civ_recency", "adjusted_form", "duration_profile", "head_to_head"},
        )

        ov = build_window_temporal_overrides(conn, visible_match_cap=30)
        out = apply_window_temporal_overrides(full.copy(), ov)
        row = out.loc[out["game_id"] == 99].iloc[0]

        # 5 visible prior games; span = day20 − day0 = 20 days (current match excluded).
        assert row["window_span_days_a"] == 20
        # Longest idle gap inside the window = day12 − day2 = 10 days.
        assert row["gap_max_window_a"] == 10
        # Density = 5 games / (20 + 1) days.
        assert row["games_per_day_window_a"] == pytest.approx(5 / 21)
        # Activity windows anchored on the current match (day 20): the day-13 game
        # sits on the inclusive 7-day edge; all 5 prior games fall in the last 30.
        assert row["wt_act_7d_a"] == 1
        assert row["wt_act_30d_a"] == 5
        # New player B has no prior games → zero recent activity.
        assert row["wt_act_30d_b"] == 0
    finally:
        conn.close()
