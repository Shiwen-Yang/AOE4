from datetime import datetime, timedelta
from pathlib import Path
import sys

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoe4_predict.db import init_schema
from aoe4_predict.features import build_civ_matchup_priors, build_player_stats, build_training_features
from aoe4_predict.features_extra import (
    build_api_cap_p1_p3_p4_p5_overrides,
    apply_api50_p1_p3_p4_p5_overrides,
    build_api50_p1_p3_p4_p5_overrides,
    extend_training_features,
)


def _insert_game(conn: duckdb.DuckDBPyConnection, game_id: int, started_at: datetime, duration: int, p1: tuple, p2: tuple) -> None:
    conn.execute(
        """
        INSERT INTO games (
            game_id, started_at, finished_at, duration, map_id, map, kind, server, patch, season, source_file
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            game_id,
            started_at,
            started_at + timedelta(seconds=duration),
            duration,
            1,
            "Dry Arabia",
            "rm_solo",
            "steam",
            "patch-test",
            10,
            "synthetic",
        ],
    )
    conn.executemany(
        """
        INSERT INTO participants (
            game_id, profile_id, result, civilization, civilization_randomized,
            rating, rating_diff, mmr, mmr_diff, input_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                game_id,
                p1[0],
                p1[1],
                p1[2],
                False,
                p1[3],
                0,
                p1[4],
                0,
                "mouse_keyboard",
            ),
            (
                game_id,
                p2[0],
                p2[1],
                p2[2],
                False,
                p2[3],
                0,
                p2[4],
                0,
                "mouse_keyboard",
            ),
        ],
    )


def test_api50_overrides_cap_p1_p3_p4_p5_and_preserve_schema():
    conn = duckdb.connect(":memory:")
    try:
        init_schema(conn)

        base_time = datetime(2026, 1, 1, 0, 0, 0)

        # Earliest mutual meeting. This should fall out of both players' visible
        # API50 histories by the time they meet again in game 102.
        _insert_game(
            conn,
            1,
            base_time,
            600,
            (1, True, "english", 1400, 1400),
            (2, False, "french", 1500, 1500),
        )

        # Player 1 then plays 50 non-overlapping games.
        for idx in range(2, 52):
            _insert_game(
                conn,
                idx,
                base_time + timedelta(hours=idx - 1),
                600 if idx % 2 == 0 else 2400,
                (1, idx % 3 != 0, "english", 1400 + idx, 1400 + idx),
                (1000 + idx, idx % 3 == 0, "abbasid", 1200, 1200),
            )

        # Player 2 then plays 50 different non-overlapping games.
        for idx in range(52, 102):
            _insert_game(
                conn,
                idx,
                base_time + timedelta(hours=idx - 1),
                600 if idx % 2 == 0 else 2400,
                (2, idx % 4 != 0, "french", 1500 + idx, 1500 + idx),
                (2000 + idx, idx % 4 == 0, "hre", 1250, 1250),
            )

        # Current target meeting.
        _insert_game(
            conn,
            102,
            base_time + timedelta(hours=101),
            1200,
            (1, False, "english", 1600, 1600),
            (2, True, "french", 1700, 1700),
        )

        build_player_stats(conn)
        build_civ_matchup_priors(conn)
        full_df = build_training_features(conn, train_seasons=[10])
        full_df = extend_training_features(
            conn,
            full_df,
            {"civ_recency", "adjusted_form", "duration_profile", "head_to_head"},
        )

        overrides = build_api50_p1_p3_p4_p5_overrides(conn)
        capped_df = apply_api50_p1_p3_p4_p5_overrides(full_df.copy(), overrides)

        full_row = full_df.loc[full_df["game_id"] == 102].iloc[0]
        capped_row = capped_df.loc[capped_df["game_id"] == 102].iloc[0]
        override_row = overrides.loc[overrides["game_id"] == 102].iloc[0]

        assert list(capped_df.columns) == list(full_df.columns)
        assert override_row["visible_games_a"] <= 50
        assert override_row["visible_games_b"] <= 50

        # P1: 60-day civ recency drops the oldest visible game under the 50-game cap.
        assert full_row["civ_games_60d_a"] == 51
        assert capped_row["civ_games_60d_a"] == 50
        assert full_row["civ_games_60d_b"] == 51
        assert capped_row["civ_games_60d_b"] == 50

        # P3: recent-form features depend only on the most recent 20 games and should match.
        assert full_row["recent_w_20_a"] == capped_row["recent_w_20_a"]
        assert full_row["recent_wr_20_a"] == capped_row["recent_wr_20_a"]
        assert full_row["recent_w_20_b"] == capped_row["recent_w_20_b"]
        assert full_row["recent_wr_20_b"] == capped_row["recent_wr_20_b"]

        # P4: lifetime-style duration shares change under the capped visible history.
        assert full_row["short_games_a"] == 26
        assert capped_row["short_games_a"] == 25
        assert full_row["short_share_a"] == pytest.approx(26 / 51)
        assert capped_row["short_share_a"] == pytest.approx(0.5)

        # P5: the only prior head-to-head falls outside both capped histories.
        assert full_row["h2h_games"] == 1
        assert capped_row["h2h_games"] == 0
        assert full_row["h2h_wins_a"] == 1
        assert capped_row["h2h_wins_a"] == 0
    finally:
        conn.close()


def test_api_cap_builder_respects_requested_cap():
    conn = duckdb.connect(":memory:")
    try:
        init_schema(conn)
        base_time = datetime(2026, 1, 1, 0, 0, 0)

        for idx in range(1, 36):
            _insert_game(
                conn,
                idx,
                base_time + timedelta(hours=idx - 1),
                900,
                (1, idx % 2 == 0, "english", 1400 + idx, 1400 + idx),
                (2000 + idx, idx % 2 != 0, "french", 1200, 1200),
            )

        _insert_game(
            conn,
            36,
            base_time + timedelta(hours=35),
            1200,
            (1, True, "english", 1500, 1500),
            (9999, False, "rus", 1200, 1200),
        )

        build_player_stats(conn)
        build_civ_matchup_priors(conn)
        build_training_features(conn, train_seasons=[10])

        cap_30 = build_api_cap_p1_p3_p4_p5_overrides(conn, visible_match_cap=30)
        row = cap_30.loc[cap_30["game_id"] == 36].iloc[0]
        assert row["visible_games_a"] == 30
        assert row["visible_games_b"] == 0
    finally:
        conn.close()
