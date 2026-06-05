"""
Load JSON.gz dump files into DuckDB games + participants tables.

Schema variation by season:
  S3: no mmr, no mmr_diff, no input_type → insert NULL
  S4+: all fields present
"""
import gzip
import json
import re
import time
from pathlib import Path

import pandas as pd

from .config import ALL_SEASONS, DATA_DIR, INGEST_BATCH_SIZE, RM_1V1_KINDS
from .db import get_conn, init_schema, row_count


def _season_from_path(path: Path) -> int:
    m = re.search(r"_s(\d+)\.json", path.name)
    if not m:
        raise ValueError(f"Cannot parse season from filename: {path.name}")
    return int(m.group(1))


def _normalize_result(val: str | None) -> bool | None:
    if val == "win":
        return True
    if val == "loss":
        return False
    return None


def _load_file(path: Path, season: int) -> tuple[list[dict], list[dict]]:
    """Return (games_rows, participants_rows) for one JSON.gz file."""
    source_file = path.name
    games_rows: list[dict] = []
    participants_rows: list[dict] = []

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        records = json.load(f)

    for rec in records:
        kind = rec.get("kind", "")
        if kind not in RM_1V1_KINDS:
            continue

        teams = rec.get("teams", [])
        flat_players = [p for team in teams for p in team]
        if len(flat_players) != 2:
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

        for player in flat_players:
            participants_rows.append(
                {
                    "game_id": rec["game_id"],
                    "profile_id": player["profile_id"],
                    "result": _normalize_result(player.get("result")),
                    "civilization": player.get("civilization"),
                    "civilization_randomized": player.get("civilization_randomized"),
                    "rating": player.get("rating"),
                    "rating_diff": player.get("rating_diff"),
                    "mmr": player.get("mmr"),           # NULL for S3
                    "mmr_diff": player.get("mmr_diff"), # NULL for S3
                    "input_type": player.get("input_type"),  # NULL for S3/S4
                }
            )

    return games_rows, participants_rows


def _insert_batch(conn, games_rows: list[dict], participants_rows: list[dict]) -> None:
    if games_rows:
        df_g = pd.DataFrame(games_rows)
        df_g["started_at"] = pd.to_datetime(df_g["started_at"], utc=True).dt.tz_localize(None)
        df_g["finished_at"] = pd.to_datetime(df_g["finished_at"], utc=True).dt.tz_localize(None)
        conn.register("_batch_games", df_g)
        conn.execute("INSERT OR IGNORE INTO games SELECT * FROM _batch_games")
        conn.unregister("_batch_games")

    if participants_rows:
        df_p = pd.DataFrame(participants_rows)
        conn.register("_batch_parts", df_p)
        conn.execute("INSERT OR IGNORE INTO participants SELECT * FROM _batch_parts")
        conn.unregister("_batch_parts")


def ingest_file(path: Path, conn, season: int | None = None) -> dict:
    """Ingest one JSON.gz file. Returns counts dict."""
    if season is None:
        season = _season_from_path(path)

    t0 = time.time()
    print(f"  Loading {path.name} (season {season})...", end=" ", flush=True)
    games_rows, participants_rows = _load_file(path, season)

    # Insert in batches to avoid memory spikes
    for i in range(0, max(len(games_rows), 1), INGEST_BATCH_SIZE):
        _insert_batch(
            conn,
            games_rows[i : i + INGEST_BATCH_SIZE],
            participants_rows[i * 2 : (i + INGEST_BATCH_SIZE) * 2],
        )

    elapsed = time.time() - t0
    print(f"{len(games_rows):,} games in {elapsed:.1f}s")
    return {"season": season, "games": len(games_rows), "participants": len(participants_rows)}


def ingest_all(
    data_dir: Path | None = None,
    db_path=None,
    seasons: list[int] | None = None,
    skip_existing: bool = True,
) -> list[dict]:
    """
    Ingest all (or specified) seasons from data_dir into DuckDB.

    Args:
        seasons: list of season numbers to ingest; defaults to all found in data_dir
        skip_existing: if True, skip seasons already present in the DB
    """
    data_dir = data_dir or DATA_DIR
    conn = get_conn(db_path)
    init_schema(conn)

    if skip_existing:
        existing = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT season FROM games WHERE season IS NOT NULL"
            ).fetchall()
        }
    else:
        existing = set()

    pattern_map: dict[int, Path] = {}
    for p in sorted(data_dir.glob("games_rm_1v1_s*.json*")):
        try:
            s = _season_from_path(p)
            pattern_map[s] = p
        except ValueError:
            pass

    target_seasons = seasons or sorted(pattern_map.keys())
    results = []

    for s in target_seasons:
        if s in existing:
            n = conn.execute(
                "SELECT count(*) FROM games WHERE season = ?", [s]
            ).fetchone()[0]
            print(f"  Season {s}: already ingested ({n:,} games), skipping.")
            continue
        if s not in pattern_map:
            print(f"  Season {s}: no file found in {data_dir}, skipping.")
            continue
        result = ingest_file(pattern_map[s], conn, season=s)
        results.append(result)

    total_games = row_count(conn, "games")
    total_parts = row_count(conn, "participants")
    print(f"\nDB totals: {total_games:,} games, {total_parts:,} participants")
    conn.close()
    return results
