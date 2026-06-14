from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import duckdb

AOE4WORLD_GAME_URL = "https://aoe4world.com/api/v0/games/{game_id}"
OFFICIAL_HISTORY_URL = (
    "https://aoe-api.worldsedgelink.com/community/leaderboard/getRecentMatchHistory"
)
USER_AGENT = "AOE4ReplayHarvest/0.1 (yangshw0223@gmail.com)"

JsonFetcher = Callable[[str], Any]


@dataclass(frozen=True)
class Outcome:
    winner_profile_id: int
    loser_profile_id: int


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

_OUTCOME_FETCHES_DDL = """
CREATE TABLE IF NOT EXISTS replay_outcome_fetches (
    game_id BIGINT,
    source VARCHAR,
    profile_id_used BIGINT,
    fetched_at TIMESTAMP,
    status VARCHAR,
    winner_profile_id BIGINT,
    loser_profile_id BIGINT,
    last_error VARCHAR,
    PRIMARY KEY (game_id, source)
)
"""


def _ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(_OUTCOME_FETCHES_DDL)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_replay_outcomes_status "
        "ON replay_outcome_fetches(status)"
    )
    # Add player_slot to participants if not already present.
    # DuckDB ALTER TABLE ADD COLUMN IF NOT EXISTS is supported.
    conn.execute(
        "ALTER TABLE participants ADD COLUMN IF NOT EXISTS player_slot INTEGER"
    )


# ---------------------------------------------------------------------------
# Replay slot extraction via inspect-full
# ---------------------------------------------------------------------------

def inspect_replay_slots(
    raw_path: Path,
    parser_project: Path,
) -> dict[int, int] | None:
    """Return {player_slot: profile_id} (slots 1 and 2) by running inspect-full.

    player_slot is determined by the order of entries in the players array:
    index 0 → slot 1, index 1 → slot 2.  This matches how IntentTimelineJsonlWriter
    assigns player_slot via TimelinePlayerSlots.FromReplay (index+1).

    Returns None if the subprocess fails or profile_ids cannot be extracted.
    """
    try:
        result = subprocess.run(
            [
                "dotnet",
                "run",
                "--project",
                str(parser_project),
                "--",
                "inspect-full",
                str(raw_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        # inspect-full may exit with code 1 even on success (partial parse warning);
        # treat stdout as authoritative if it contains valid JSON.
        if not result.stdout.strip():
            return None
        data = json.loads(result.stdout)
    except Exception:
        return None

    players = (
        data.get("metadata") or {}
    ).get("game_setup") or {}
    players = players.get("players") or []

    if len(players) < 2:
        return None

    slot_map: dict[int, int] = {}
    for slot_index, player in enumerate(players):
        slot = slot_index + 1
        raw_profile_id = player.get("profile_id")
        if not raw_profile_id:
            return None
        try:
            slot_map[slot] = int(raw_profile_id)
        except (ValueError, TypeError):
            return None

    return slot_map if len(slot_map) >= 2 else None


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def fetch_json(url: str) -> Any:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# AoE4World game endpoint parser
# ---------------------------------------------------------------------------

def _flat_players(game: dict[str, Any]) -> list[dict[str, Any]]:
    players: list[dict[str, Any]] = []
    for team in game.get("teams") or []:
        for entry in team or []:
            player = entry.get("player") if isinstance(entry, dict) else entry
            if isinstance(player, dict):
                players.append(player)
            elif isinstance(entry, dict) and "profile_id" in entry:
                players.append(entry)
    return players


def parse_aoe4world_game(
    payload: Any,
    game_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], Outcome | None]:
    """Parse AoE4World /api/v0/games/{id} response.

    Returns (game_row, participant_rows, outcome).
    participant_rows each have: profile_id, result (bool|None), civilization,
    civilization_randomized, rating, rating_diff, mmr, mmr_diff, input_type.
    """
    if not isinstance(payload, dict):
        return {}, [], None

    # Unwrap if nested under a key
    game = payload
    if "game_id" not in payload:
        for key in ("game", "data"):
            if isinstance(payload.get(key), dict):
                game = payload[key]
                break

    if game.get("game_id") is not None and int(game["game_id"]) != game_id:
        return {}, [], None

    game_row: dict[str, Any] = {
        "game_id": game_id,
        "started_at": game.get("started_at"),
        "map": game.get("map"),
        "kind": game.get("kind") or "rm_1v1",
        "season": game.get("season"),
        "patch": game.get("patch"),
        "duration": game.get("duration"),
        "server": game.get("server"),
        "source_file": f"aoe4world_api:{game_id}",
    }

    players = _flat_players(game)
    participant_rows: list[dict[str, Any]] = []
    winner_id: int | None = None
    loser_id: int | None = None

    for player in players:
        profile_id = player.get("profile_id")
        if profile_id is None:
            continue
        profile_id = int(profile_id)
        result_str = player.get("result")
        if result_str == "win":
            result_bool: bool | None = True
            winner_id = profile_id
        elif result_str == "loss":
            result_bool = False
            loser_id = profile_id
        else:
            result_bool = None

        participant_rows.append(
            {
                "profile_id": profile_id,
                "result": result_bool,
                "civilization": player.get("civilization"),
                "civilization_randomized": player.get("civilization_randomized"),
                "rating": player.get("rating"),
                "rating_diff": player.get("rating_diff"),
                "mmr": player.get("mmr"),
                "mmr_diff": player.get("mmr_diff"),
                "input_type": player.get("input_type"),
            }
        )

    outcome: Outcome | None = None
    if winner_id is not None and loser_id is not None:
        outcome = Outcome(winner_profile_id=winner_id, loser_profile_id=loser_id)

    return game_row, participant_rows, outcome


# ---------------------------------------------------------------------------
# Official API fallback parser
# ---------------------------------------------------------------------------

def parse_official_outcome(payload: Any, game_id: int) -> Outcome | None:
    if not isinstance(payload, dict):
        return None
    reports = payload.get("matchhistoryreportresults") or []
    if not isinstance(reports, list):
        return None

    rows: list[tuple[int, bool]] = []
    for report in reports:
        if not isinstance(report, dict):
            continue
        match_id = report.get("matchhistory_id")
        if match_id is not None and int(match_id) != game_id:
            continue
        profile_id = report.get("profile_id")
        resulttype = report.get("resulttype")
        if profile_id is None or resulttype is None:
            continue
        if int(resulttype) == 1:
            rows.append((int(profile_id), True))
        elif int(resulttype) == 0:
            rows.append((int(profile_id), False))

    winners = [pid for pid, r in rows if r]
    losers = [pid for pid, r in rows if not r]
    if len(winners) == 1 and len(losers) == 1:
        return Outcome(winner_profile_id=winners[0], loser_profile_id=losers[0])
    return None


# ---------------------------------------------------------------------------
# Game selection
# ---------------------------------------------------------------------------

def games_needing_outcomes(
    conn: duckdb.DuckDBPyConnection,
    parsed_dir: Path,
    limit: int,
) -> list[int]:
    """Return game_ids that have a parsed JSONL but no complete outcome in the DB."""
    import re

    pattern = re.compile(r"^replay_(\d+)\.intent_timeline\.jsonl$")
    all_ids = []
    for path in parsed_dir.iterdir():
        m = pattern.match(path.name)
        if m:
            all_ids.append(int(m.group(1)))
    all_ids.sort()

    if not all_ids:
        return []

    # Find which game_ids are already fully labeled (2 participants, both non-NULL result)
    ids_str = ",".join(str(g) for g in all_ids)
    labeled = set(
        row[0]
        for row in conn.execute(
            f"""
            SELECT p.game_id
            FROM participants p
            WHERE p.game_id IN ({ids_str})
            GROUP BY p.game_id
            HAVING count(*) = 2
               AND count(CASE WHEN p.result IS NOT NULL THEN 1 END) = 2
            """
        ).fetchall()
    )

    result = [g for g in all_ids if g not in labeled]
    return result[:limit]


# ---------------------------------------------------------------------------
# DB writes
# ---------------------------------------------------------------------------

def _find_raw_path(raw_dir: Path, game_id: int) -> Path | None:
    for path in raw_dir.rglob(f"replay_{game_id}"):
        return path
    return None


def apply_game_and_outcome(
    conn: duckdb.DuckDBPyConnection,
    game_id: int,
    game_row: dict[str, Any],
    participant_rows: list[dict[str, Any]],
    slot_map: dict[int, int] | None,
    raw_path: Path | None,
) -> str:
    if not participant_rows:
        return "no_participants"

    # Build reverse map: profile_id → player_slot
    profile_to_slot: dict[int, int] = {}
    if slot_map:
        for slot_num, pid in slot_map.items():
            profile_to_slot[pid] = slot_num

    # Check existing DB state for conflict detection only
    existing = {
        int(row[0]): (row[1], row[2])  # profile_id → (result, player_slot)
        for row in conn.execute(
            "SELECT profile_id, result, player_slot FROM participants WHERE game_id = ?",
            [game_id],
        ).fetchall()
    }
    incoming_pids = {int(r["profile_id"]) for r in participant_rows}

    # Conflict: existing profile_ids differ from incoming
    if existing and set(existing) != incoming_pids:
        return "conflict"

    # Check if fully labeled (result + player_slot already set correctly)
    if existing and set(existing) == incoming_pids:
        def _matches(r: dict[str, Any]) -> bool:
            pid = int(r["profile_id"])
            db_result, db_slot = existing[pid]
            expected_slot = profile_to_slot.get(pid)
            return db_result == r["result"] and (expected_slot is None or db_slot == expected_slot)
        if all(_matches(r) for r in participant_rows):
            return "already_labeled"

    # Insert game row if not present
    if game_row.get("game_id") is not None:
        conn.execute(
            """
            INSERT OR IGNORE INTO games
                (game_id, started_at, map, kind, season, patch, duration, server, source_file)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                game_row.get("game_id"),
                game_row.get("started_at"),
                game_row.get("map"),
                game_row.get("kind"),
                game_row.get("season"),
                game_row.get("patch"),
                game_row.get("duration"),
                game_row.get("server"),
                game_row.get("source_file"),
            ],
        )

    # Insert/update participants
    for row in participant_rows:
        pid = int(row["profile_id"])
        slot = profile_to_slot.get(pid)
        conn.execute(
            """
            INSERT INTO participants
                (game_id, profile_id, result, civilization, civilization_randomized,
                 rating, rating_diff, mmr, mmr_diff, input_type, player_slot)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (game_id, profile_id) DO UPDATE SET
                result = excluded.result,
                civilization = coalesce(excluded.civilization, participants.civilization),
                player_slot = coalesce(excluded.player_slot, participants.player_slot)
            """,
            [
                game_id,
                pid,
                row.get("result"),
                row.get("civilization"),
                row.get("civilization_randomized"),
                row.get("rating"),
                row.get("rating_diff"),
                row.get("mmr"),
                row.get("mmr_diff"),
                row.get("input_type"),
                slot,
            ],
        )

    # Register in replay_downloads if not already there
    if raw_path is not None:
        conn.execute(
            """
            INSERT OR IGNORE INTO replay_downloads
                (game_id, raw_path, status, source, sample_group, downloaded_at)
            VALUES (?, ?, 'downloaded', 'external_replay', 'realtime_replay', ?)
            """,
            [game_id, str(raw_path), datetime.now(timezone.utc)],
        )

    return "filled"


def _record_fetch(
    conn: duckdb.DuckDBPyConnection,
    game_id: int,
    source: str,
    profile_id_used: int | None,
    status: str,
    outcome: Outcome | None,
    error: str | None,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO replay_outcome_fetches
            (game_id, source, profile_id_used, fetched_at, status,
             winner_profile_id, loser_profile_id, last_error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            game_id,
            source,
            profile_id_used,
            datetime.now(timezone.utc),
            status,
            outcome.winner_profile_id if outcome else None,
            outcome.loser_profile_id if outcome else None,
            error,
        ],
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def hydrate_outcomes(
    conn: duckdb.DuckDBPyConnection,
    parsed_dir: Path,
    raw_dir: Path,
    parser_project: Path,
    limit: int = 500,
    sleep_seconds: float = 1.0,
    fetcher: JsonFetcher = fetch_json,
    use_official_fallback: bool = True,
) -> dict[str, int]:
    _ensure_schema(conn)

    counts: dict[str, int] = {
        "filled": 0,
        "already_labeled": 0,
        "unresolved": 0,
        "conflict": 0,
        "failed": 0,
        "no_slot_map": 0,
        "no_participants": 0,
    }

    game_ids = games_needing_outcomes(conn, parsed_dir, limit)
    print(f"Games needing outcomes: {len(game_ids)}")

    for game_id in game_ids:
        raw_path = _find_raw_path(raw_dir, game_id)
        slot_map: dict[int, int] | None = None
        if raw_path is not None:
            slot_map = inspect_replay_slots(raw_path, parser_project)

        # --- Primary: AoE4World game endpoint ---
        source = "aoe4world"
        profile_id_used = next(iter(slot_map.values())) if slot_map else None
        outcome: Outcome | None = None
        status = "failed"
        error: str | None = None

        try:
            url = AOE4WORLD_GAME_URL.format(game_id=game_id)
            payload = fetcher(url)
            game_row, participant_rows, outcome = parse_aoe4world_game(payload, game_id)

            if not participant_rows:
                raise ValueError("no participants in AoE4World response")

            status = apply_game_and_outcome(
                conn, game_id, game_row, participant_rows, slot_map, raw_path
            )

        except Exception as exc:
            error = str(exc)

            # --- Fallback: official World's Edge API ---
            if use_official_fallback and slot_map:
                fallback_profile_id = next(iter(slot_map.values()))
                source = "official_recent_match_history"
                profile_id_used = fallback_profile_id
                try:
                    params = urlencode(
                        {"title": "age4", "profile_ids": f"[{fallback_profile_id}]"}
                    )
                    fallback_url = f"{OFFICIAL_HISTORY_URL}?{params}"
                    payload = fetcher(fallback_url)
                    outcome = parse_official_outcome(payload, game_id)
                    if outcome is not None:
                        # Write result-only update (game row may be missing — acceptable)
                        for pid, result in [
                            (outcome.winner_profile_id, True),
                            (outcome.loser_profile_id, False),
                        ]:
                            slot = slot_map and {v: k for k, v in slot_map.items()}.get(pid)
                            conn.execute(
                                """
                                INSERT INTO participants (game_id, profile_id, result, player_slot)
                                VALUES (?, ?, ?, ?)
                                ON CONFLICT (game_id, profile_id) DO UPDATE SET
                                    result = excluded.result,
                                    player_slot = coalesce(excluded.player_slot, participants.player_slot)
                                """,
                                [game_id, pid, result, slot],
                            )
                        status = "filled"
                        error = None
                    else:
                        status = "unresolved"
                        error = "outcome_not_found"
                except Exception as fb_exc:
                    status = "failed"
                    error = f"fallback: {fb_exc}"
            elif not slot_map and use_official_fallback:
                status = "unresolved"
                error = f"no_slot_map; primary: {error}"
            else:
                status = "unresolved" if "404" in str(error) else "failed"

        _record_fetch(conn, game_id, source, profile_id_used, status, outcome, error)
        counts[status] = counts.get(status, 0) + 1

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return counts
