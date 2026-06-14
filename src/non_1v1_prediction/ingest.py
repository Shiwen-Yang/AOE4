"""
Load TEAM-mode JSON.gz dumps into aoe4_team.duckdb (games + participants).

Generic over team size: each player is stored with its `team_id` (the index of its
team in the record's `teams` array). Only games with exactly 2 complete teams of the
expected size are kept (e.g. 4v4 → two teams of 4). The JSON schema is identical to
the 1v1 dumps; only the team structure differs.
"""
import gzip
import json
import re
import time
from pathlib import Path

import pandas as pd

from .config import INGEST_BATCH_SIZE, MODE_TEAM_SIZE, TEAM_DATA_DIR
from .db import get_conn, init_schema, row_count


def _season_from_path(path: Path) -> int:
    m = re.search(r"_s(\d+)\.json", path.name)
    if not m:
        raise ValueError(f"Cannot parse season from filename: {path.name}")
    return int(m.group(1))


def _mode_from_path(path: Path) -> str:
    m = re.search(r"games_(rm_\d+v\d+)_s\d+", path.name)
    if not m:
        raise ValueError(f"Cannot parse mode from filename: {path.name}")
    return m.group(1)


def _normalize_result(val: str | None) -> bool | None:
    if val == "win":
        return True
    if val == "loss":
        return False
    return None


def _load_file(path: Path, season: int, mode: str) -> tuple[list[dict], list[dict]]:
    """Return (games_rows, participants_rows) for one team-mode JSON.gz file."""
    source_file = path.name
    team_size = MODE_TEAM_SIZE.get(mode)
    games_rows: list[dict] = []
    participants_rows: list[dict] = []

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        records = json.load(f)

    for rec in records:
        kind = rec.get("kind", "")
        if kind != mode:
            continue

        teams = rec.get("teams", []) or []
        # Require exactly 2 complete teams of the expected size.
        if len(teams) != 2:
            continue
        if team_size is not None and any(len(t) != team_size for t in teams):
            continue

        games_rows.append(
            {
                "game_id": rec["game_id"],
                "started_at": rec.get("started_at"),
                "finished_at": rec.get("finished_at"),
                "duration": rec.get("duration"),
                "map_id": rec.get("map_id"),
                "map": rec.get("map"),
                "kind": kind,
                "server": rec.get("server"),
                "patch": rec.get("patch"),
                "season": season,
                "source_file": source_file,
            }
        )

        for team_idx, team in enumerate(teams):
            for player in team:
                participants_rows.append(
                    {
                        "game_id": rec["game_id"],
                        "profile_id": player["profile_id"],
                        "team_id": team_idx,
                        "result": _normalize_result(player.get("result")),
                        "civilization": player.get("civilization"),
                        "civilization_randomized": player.get("civilization_randomized"),
                        "rating": player.get("rating"),
                        "rating_diff": player.get("rating_diff"),
                        "mmr": player.get("mmr"),
                        "mmr_diff": player.get("mmr_diff"),
                        "input_type": player.get("input_type"),
                    }
                )

    return games_rows, participants_rows


def _insert_batch(conn, games_rows: list[dict], participants_rows: list[dict]) -> None:
    if games_rows:
        df_g = pd.DataFrame(games_rows)
        df_g["started_at"] = pd.to_datetime(df_g["started_at"], utc=True).dt.tz_localize(None)
        df_g["finished_at"] = pd.to_datetime(df_g["finished_at"], utc=True).dt.tz_localize(None)
        conn.register("_batch_games", df_g)
        conn.execute("""
            INSERT INTO games
            SELECT bg.* FROM _batch_games bg
            WHERE NOT EXISTS (SELECT 1 FROM games g WHERE g.game_id = bg.game_id)
        """)
        conn.unregister("_batch_games")

    if participants_rows:
        df_p = pd.DataFrame(participants_rows)
        conn.register("_batch_parts", df_p)
        conn.execute("""
            INSERT INTO participants
                (game_id, profile_id, team_id, result, civilization,
                 civilization_randomized, rating, rating_diff, mmr, mmr_diff, input_type)
            SELECT
                bp.game_id, bp.profile_id, bp.team_id, bp.result, bp.civilization,
                bp.civilization_randomized, bp.rating, bp.rating_diff,
                bp.mmr, bp.mmr_diff, bp.input_type
            FROM _batch_parts bp
            WHERE NOT EXISTS (
                SELECT 1 FROM participants p
                WHERE p.game_id = bp.game_id AND p.profile_id = bp.profile_id
            )
        """)
        conn.unregister("_batch_parts")


def ingest_file(path: Path, conn, season: int | None = None, mode: str | None = None) -> dict:
    """Ingest one team-mode JSON.gz file. Returns counts dict."""
    season = season if season is not None else _season_from_path(path)
    mode = mode or _mode_from_path(path)

    t0 = time.time()
    print(f"  Loading {path.name} (mode {mode}, season {season})...", end=" ", flush=True)
    games_rows, participants_rows = _load_file(path, season, mode)

    per_game = MODE_TEAM_SIZE.get(mode, 4) * 2  # participants per kept game
    for i in range(0, max(len(games_rows), 1), INGEST_BATCH_SIZE):
        _insert_batch(
            conn,
            games_rows[i : i + INGEST_BATCH_SIZE],
            participants_rows[i * per_game : (i + INGEST_BATCH_SIZE) * per_game],
        )

    elapsed = time.time() - t0
    print(f"{len(games_rows):,} games in {elapsed:.1f}s")
    return {"season": season, "mode": mode, "games": len(games_rows),
            "participants": len(participants_rows)}


def ingest_all(
    mode: str = "rm_4v4",
    seasons: list[int] | None = None,
    data_dir: Path | None = None,
    db_path=None,
    skip_existing: bool = True,
) -> list[dict]:
    """Ingest the requested team-mode seasons from data_dir into aoe4_team.duckdb."""
    data_dir = Path(data_dir or TEAM_DATA_DIR)
    conn = get_conn(db_path)
    init_schema(conn)

    if skip_existing:
        existing = {
            (r[0], r[1])
            for r in conn.execute(
                "SELECT DISTINCT kind, season FROM games WHERE season IS NOT NULL"
            ).fetchall()
        }
    else:
        existing = set()

    pattern_map: dict[int, Path] = {}
    for p in sorted(data_dir.glob(f"games_{mode}_s*.json*")):
        try:
            pattern_map[_season_from_path(p)] = p
        except ValueError:
            pass

    target_seasons = seasons or sorted(pattern_map.keys())
    results = []

    for s in target_seasons:
        if (mode, s) in existing:
            n = conn.execute(
                "SELECT count(*) FROM games WHERE kind = ? AND season = ?", [mode, s]
            ).fetchone()[0]
            print(f"  {mode} season {s}: already ingested ({n:,} games), skipping.")
            continue
        if s not in pattern_map:
            print(f"  {mode} season {s}: no file found in {data_dir}, skipping.")
            continue
        results.append(ingest_file(pattern_map[s], conn, season=s, mode=mode))

    total_games = row_count(conn, "games")
    total_parts = row_count(conn, "participants")
    print(f"\nDB totals: {total_games:,} games, {total_parts:,} participants")
    conn.close()
    return results
