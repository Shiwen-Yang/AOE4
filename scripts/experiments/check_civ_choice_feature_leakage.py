"""Leakage checks for civ-choice recent-history features.

This samples player-matches, recomputes v2 sequence features from raw history
using only games strictly before the target timestamp, and compares them with
the feature loader output.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import sys

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.experiments.run_civ_choice_intent_features import (
    RANDOM_CIV,
    _load_matrix,
)


OUT_PATH = ROOT / "reports/generated/civ_choice_v2_leakage_audit.json"


def _entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return float(
        -sum((n / total) * math.log(n / total) for n in counts.values() if n > 0)
    )


def _history_for(conn: duckdb.DuckDBPyConnection, profile_id: int, started_at) -> list[dict]:
    return conn.execute(
        """
        SELECT
            p.game_id,
            CASE
                WHEN p.civilization_randomized = TRUE THEN ?
                ELSE p.civilization
            END AS civ,
            g.started_at,
            g.map
        FROM participants p
        JOIN games g ON p.game_id = g.game_id
        WHERE p.profile_id = ?
          AND g.kind IN ('rm_1v1', 'rm_solo')
          AND p.result IS NOT NULL
          AND p.civilization IS NOT NULL
          AND g.started_at < ?
        ORDER BY g.started_at DESC, p.game_id DESC
        LIMIT 20
        """,
        [RANDOM_CIV, int(profile_id), started_at],
    ).df().to_dict("records")


def main() -> None:
    conn = duckdb.connect(str(ROOT / "aoe4.duckdb"), read_only=True)
    try:
        df = _load_matrix(conn, hash_mod=200, hash_remainder=0)
        groups = (
            df[["game_id", "profile_id", "started_at", "map"]]
            .drop_duplicates(["game_id", "profile_id"])
            .head(200)
        )

        mismatches = []
        checked_groups = 0
        checked_rows = 0
        current_game_leak_count = 0

        for group in groups.itertuples(index=False):
            hist = _history_for(conn, group.profile_id, group.started_at)
            civs20 = [str(r["civ"]) for r in hist]
            maps20 = [str(r["map"]) for r in hist]
            civs10 = civs20[:10]
            counts = {
                1: Counter(civs20[:1]),
                2: Counter(civs20[:2]),
                3: Counter(civs20[:3]),
                5: Counter(civs20[:5]),
                10: Counter(civs10),
            }
            same_map_mask = [m == str(group.map) for m in maps20]
            same_map_counts = Counter(
                civ for civ, same_map in zip(civs20, same_map_mask) if same_map
            )
            player_same_map = sum(same_map_mask)
            last_pos = {}
            for i, civ in enumerate(civs20, start=1):
                last_pos.setdefault(civ, i)
            streak_len = 0
            if civs20:
                for civ in civs20:
                    if civ != civs20[0]:
                        break
                    streak_len += 1
            switch_count = sum(
                1 for i in range(1, len(civs10)) if civs10[i] != civs10[i - 1]
            )
            unique10 = len(counts[10])
            entropy10 = _entropy(counts[10])

            sub = df[(df["game_id"] == group.game_id) & (df["profile_id"] == group.profile_id)]
            target_civ = sub.loc[sub["target"] == 1, "chosen_civ"].astype(str).iloc[0]
            if hist and int(hist[0]["game_id"]) == int(group.game_id):
                current_game_leak_count += 1

            for row in sub.itertuples(index=False):
                civ = str(row.candidate_civ)
                expected = {
                    "cand_games_last_1_games": counts[1].get(civ, 0),
                    "cand_games_last_2_games": counts[2].get(civ, 0),
                    "cand_games_last_3_games": counts[3].get(civ, 0),
                    "cand_games_last_5_games": counts[5].get(civ, 0),
                    "cand_games_last_10_games": counts[10].get(civ, 0),
                    "candidate_last_played_position": last_pos.get(civ, 21),
                    "player_current_streak_len": streak_len,
                    "candidate_current_streak_len": streak_len if civs20 and civ == civs20[0] else 0,
                    "recent_civ_switch_count_last_10_games": switch_count,
                    "recent_unique_civs_last_10_games": unique10,
                    "recent_entropy_last_10_games": entropy10,
                    "cand_games_last_20_same_map": same_map_counts.get(civ, 0),
                    "player_games_last_20_same_map": player_same_map,
                }
                for col, value in expected.items():
                    actual = getattr(row, col)
                    if isinstance(value, float):
                        ok = np.isclose(float(actual), value, atol=1e-6)
                    else:
                        ok = int(actual) == int(value)
                    if not ok:
                        mismatches.append(
                            {
                                "game_id": int(group.game_id),
                                "profile_id": int(group.profile_id),
                                "candidate_civ": civ,
                                "target_civ": target_civ,
                                "column": col,
                                "actual": float(actual) if isinstance(actual, float) else int(actual),
                                "expected": value,
                            }
                        )
                        if len(mismatches) >= 20:
                            break
                checked_rows += 1
                if len(mismatches) >= 20:
                    break
            checked_groups += 1
            if len(mismatches) >= 20:
                break
    finally:
        conn.close()

    dataset_sql = (ROOT / "src/civ_choice/dataset.py").read_text()
    static_checks = {
        "random_civ_in_dataset": RANDOM_CIV in dataset_sql,
        "strict_history_filter_present": "started_at < cr.started_at" in dataset_sql,
        "no_randomized_filter_in_player_raw_games": "p.civilization_randomized = FALSE" not in dataset_sql,
    }
    result = {
        "sample": {"hash_mod": 200, "hash_remainder": 0},
        "checked_groups": checked_groups,
        "checked_rows": checked_rows,
        "current_game_leak_count": current_game_leak_count,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "static_checks": static_checks,
        "passed": (
            len(mismatches) == 0
            and current_game_leak_count == 0
            and all(static_checks.values())
        ),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
