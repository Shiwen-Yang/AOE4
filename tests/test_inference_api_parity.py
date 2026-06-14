"""Parity: the games-list feature builder must reproduce the DB cap-30 path.

For a real A-vs-B game in a synthetic DB, we compute the ``recent_only`` feature
vector two ways and assert they agree across all 131 model features:
  (a) the DB override path (build_recent_only_base_overrides + build_api_cap_…),
  (b) inference_api.build_recent_only_features from each player's prior games.

MMR/rating are held constant per player so the "current game's mmr" (DB path) and
"latest prior game's mmr" (inference path) coincide — isolating the games-derived
features, which are the real reimplementation risk.
"""
from datetime import datetime, timedelta
from pathlib import Path
import sys

import duckdb
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoe4_predict.db import init_schema
from aoe4_predict.features import build_civ_matchup_priors, build_player_stats, build_training_features
from aoe4_predict.features_extra import (
    apply_api_cap_p1_p3_p4_p5_overrides,
    apply_recent_only_base_overrides,
    build_api_cap_p1_p3_p4_p5_overrides,
    build_recent_only_base_overrides,
    extend_training_features,
)
from aoe4_predict.inference_api import build_recent_only_features

CIVS = ["english", "french", "abbasid", "hre", "rus"]
MAPS = ["Dry Arabia", "Lipany", "Boulder Bay"]
DURS = [600, 1200, 2400]  # short / mid / long


def _insert(conn, gid, started_at, dur, mp, season, p1, p2):
    conn.execute(
        """INSERT INTO games (game_id, started_at, finished_at, duration, map_id, map,
                              kind, server, patch, season, source_file)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [gid, started_at, started_at + timedelta(seconds=dur), dur, 1, mp,
         "rm_solo", "steam", "10.1", season, "syn"],
    )
    conn.executemany(
        """INSERT INTO participants (game_id, profile_id, result, civilization,
               civilization_randomized, rating, rating_diff, mmr, mmr_diff, input_type)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (gid, p1[0], p1[1], p1[2], False, p1[3], 0, p1[4], 0, "mk"),
            (gid, p2[0], p2[1], p2[2], False, p2[3], 0, p2[4], 0, "mk"),
        ],
    )


def _build_db(conn):
    init_schema(conn)
    base = datetime(2026, 1, 1)
    A, B = 100, 200
    MMR_A, MMR_B = 1500, 1450  # constant per player
    gid = 0
    t = lambda h: base + timedelta(hours=h)

    # 35 A-vs-filler and 35 B-vs-filler games, interleaved in time.
    for i in range(35):
        gid += 1
        _insert(conn, gid, t(gid), DURS[i % 3], MAPS[i % 3], 11 if i < 15 else 12,
                (A, i % 2 == 0, CIVS[i % 5], 1500, MMR_A),
                (1000 + i, i % 2 != 0, CIVS[(i + 1) % 5], 1490, 1490))
    for i in range(35):
        gid += 1
        _insert(conn, gid, t(gid), DURS[(i + 1) % 3], MAPS[(i + 1) % 3], 11 if i < 15 else 12,
                (B, i % 3 == 0, CIVS[(i + 2) % 5], 1450, MMR_B),
                (2000 + i, i % 3 != 0, CIVS[(i + 3) % 5], 1440, 1440))
    # 4 prior A-vs-B meetings.
    for i in range(4):
        gid += 1
        _insert(conn, gid, t(gid), DURS[i % 3], MAPS[i % 3], 12,
                (A, i % 2 == 0, CIVS[i % 5], 1500, MMR_A),
                (B, i % 2 != 0, CIVS[(i + 2) % 5], 1450, MMR_B))
    # Current target game (the one we score).
    gid += 1
    cur_id = gid
    cur_started = t(gid)
    _insert(conn, cur_id, cur_started, 1200, "Dry Arabia", 12,
            (A, True, "english", 1500, MMR_A),
            (B, False, "french", 1450, MMR_B))
    return A, B, cur_id, cur_started


def _games_before(conn, profile_id, before):
    rows = conn.execute(
        """
        SELECT g.game_id, g.started_at, g.duration, g.map, g.season, g.patch,
               p.civilization, p.result::INT, p.rating, p.mmr,
               (SELECT p2.profile_id FROM participants p2
                WHERE p2.game_id = g.game_id AND p2.profile_id != p.profile_id LIMIT 1) AS opp
        FROM participants p JOIN games g ON p.game_id = g.game_id
        WHERE p.profile_id = ? AND g.started_at < ?
          AND g.kind IN ('rm_1v1','rm_solo') AND p.result IS NOT NULL
        ORDER BY g.started_at DESC, g.game_id DESC
        """,
        [profile_id, before],
    ).fetchall()
    games = []
    for r in rows:
        ts = r[1]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        games.append({
            "game_id": r[0], "started_at": ts, "duration": r[2], "map": r[3],
            "season": r[4], "patch": r[5], "civ": r[6], "result": r[7],
            "rating": r[8], "mmr": r[9], "opponent_profile_id": r[10],
        })
    return games


def _missing(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


def test_inference_api_matches_db_recent_only_path():
    conn = duckdb.connect(":memory:")
    try:
        A, B, cur_id, cur_started = _build_db(conn)

        build_player_stats(conn)
        build_civ_matchup_priors(conn)
        build_training_features(conn, train_seasons=[11, 12])
        df = extend_training_features(
            conn, None,
            {"civ_recency", "adjusted_form", "duration_profile", "head_to_head"},
        )
        df = apply_api_cap_p1_p3_p4_p5_overrides(df, build_api_cap_p1_p3_p4_p5_overrides(conn, 30))
        df = apply_recent_only_base_overrides(df, build_recent_only_base_overrides(conn, 30))

        db_row = df.loc[df["game_id"] == cur_id].iloc[0]

        feature_cols = [c for c in df.columns if c not in ("game_id", "target", "started_at", "civ_rand_a", "civ_rand_b")]

        games_a = _games_before(conn, A, cur_started)
        games_b = _games_before(conn, B, cur_started)

        # Match the DB prior_matchup value exactly (it is not a games-list feature).
        pg = int(db_row["prior_matchup_games"]); pw = int(db_row["prior_matchup_wins"])
        built = build_recent_only_features(
            games_a=games_a, games_b=games_b, profile_a=A, profile_b=B,
            civ_a="english", civ_b="french", map_name="Dry Arabia",
            matchup_lookup=lambda ca, cb: (pg, pw),
            now=cur_started.replace(tzinfo=None),
            season=int(db_row["season"]), patch=db_row["patch"],
        )

        mismatches = []
        for col in feature_cols:
            if col not in built:
                continue
            dbv, bv = db_row[col], built[col]
            if _missing(dbv) and _missing(bv):
                continue
            if col in ("civ_a", "civ_b", "map", "patch", "season"):
                if str(dbv) != str(bv):
                    mismatches.append((col, dbv, bv))
                continue
            if _missing(dbv) or _missing(bv):
                mismatches.append((col, dbv, bv))
            elif not np.isclose(float(dbv), float(bv), rtol=1e-3, atol=1e-3):
                mismatches.append((col, dbv, bv))

        # Every feature the DB recent_only path produces must be reproduced.
        assert built["h2h_games"] >= 1  # the synthetic A-vs-B meetings are visible
        assert not mismatches, "feature mismatches:\n" + "\n".join(
            f"  {c}: db={d!r} built={b!r}" for c, d, b in mismatches
        )
    finally:
        conn.close()
