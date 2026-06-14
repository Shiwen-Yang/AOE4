import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROFILE_ID_RE = re.compile(rb"(?<!\d):(\d{5,12})(?!\d)")


@dataclass(frozen=True)
class MatchLabel:
    replay_id: int
    target: int
    slot_profile_ids: dict[int, int]
    metadata: dict[str, Any]


def _columns(conn, table: str) -> set[str]:
    try:
        return {row[0] for row in conn.execute(f"DESCRIBE {table}").fetchall()}
    except Exception:
        return set()


def _raw_path(raw_dir: Path, replay_id: int) -> Path | None:
    name = f"replay_{replay_id}"
    for path in raw_dir.glob(f"*/{name}"):
        return path
    direct = raw_dir / name
    return direct if direct.exists() else None


def _command_stream_paths(parsed_dir: Path, replay_id: int) -> list[Path]:
    names = [
        f"command_stream_replay_{replay_id}.jsonl",
        f"command_stream_AgeIV_Replay_{replay_id}.jsonl",
    ]
    paths = [parsed_dir / name for name in names]
    paths.extend(parsed_dir.glob(f"*/command_stream_*{replay_id}*.jsonl"))
    return [path for path in paths if path.exists()]


def _slot_by_internal_player_id(parsed_dir: Path, replay_id: int) -> dict[int, int]:
    """Read parser --raw command stream sidecars: internal replay player_id -> slot."""
    mapping: dict[int, int] = {}
    for path in _command_stream_paths(parsed_dir, replay_id):
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                player_slot = row.get("player_slot")
                player_id = row.get("player_id")
                if player_slot in (1, 2) and isinstance(player_id, int) and player_id > 0:
                    mapping[player_id] = int(player_slot)
                if len(mapping) >= 2:
                    return mapping
    return mapping


def _profile_order_from_raw(raw_dir: Path, replay_id: int, participant_ids: set[int]) -> list[int]:
    path = _raw_path(raw_dir, replay_id)
    if not path:
        return []
    data = path.read_bytes()
    found: list[int] = []
    seen: set[int] = set()
    for match in PROFILE_ID_RE.finditer(data):
        profile_id = int(match.group(1))
        if profile_id in participant_ids and profile_id not in seen:
            found.append(profile_id)
            seen.add(profile_id)
    return found


class LabelResolver:
    """Resolve slot-1 winner labels without guessing arbitrary slot order."""

    def __init__(
        self,
        conn,
        raw_dir: Path,
        parsed_dir: Path | None = None,
        allow_profile_id_fallback: bool = False,
    ):
        self.conn = conn
        self.raw_dir = raw_dir
        self.parsed_dir = parsed_dir
        self.allow_profile_id_fallback = allow_profile_id_fallback
        self.game_cols = _columns(conn, "games") if conn is not None else set()
        self.participant_cols = _columns(conn, "participants") if conn is not None else set()

    def resolve(self, replay_id: int) -> tuple[MatchLabel | None, str | None]:
        if self.conn is None:
            return None, "no_db"
        if not self.game_cols or not self.participant_cols:
            return None, "missing_db_tables"

        game = self._game_metadata(replay_id)
        participants = self._participants(replay_id)
        if len(participants) != 2:
            return None, "not_two_participants"

        slot_map = self._slot_map(replay_id, participants)
        if not slot_map or 1 not in slot_map:
            return None, "slot_mapping_unresolved"

        results = {int(row["profile_id"]): row.get("result") for row in participants}
        slot1_profile = slot_map[1]
        if slot1_profile not in results or results[slot1_profile] is None:
            return None, "missing_result"
        target = int(results[slot1_profile])
        return MatchLabel(
            replay_id=replay_id,
            target=target,
            slot_profile_ids=slot_map,
            metadata=game,
        ), None

    def _game_metadata(self, replay_id: int) -> dict[str, Any]:
        cols = [c for c in ("game_id", "started_at", "map", "patch", "season", "duration") if c in self.game_cols]
        if not cols:
            return {}
        row = self.conn.execute(
            f"SELECT {', '.join(cols)} FROM games WHERE game_id = ? LIMIT 1",
            [replay_id],
        ).fetchone()
        if not row:
            return {}
        return dict(zip(cols, row))

    def _participants(self, replay_id: int) -> list[dict[str, Any]]:
        cols = [
            c
            for c in (
                "profile_id",
                "player_id",
                "full_player_id",
                "result",
                "player_slot",
                "slot",
                "team",
                "civilization",
                "rating",
                "mmr",
            )
            if c in self.participant_cols
        ]
        if "profile_id" not in cols or "result" not in cols:
            return []
        rows = self.conn.execute(
            f"SELECT {', '.join(cols)} FROM participants WHERE game_id = ?",
            [replay_id],
        ).fetchall()
        return [dict(zip(cols, row)) for row in rows]

    def _slot_map(self, replay_id: int, participants: list[dict[str, Any]]) -> dict[int, int] | None:
        for col in ("player_slot", "slot"):
            if col in self.participant_cols and all(row.get(col) is not None for row in participants):
                return {int(row[col]): int(row["profile_id"]) for row in participants}

        if self.parsed_dir is not None:
            slot_by_internal = _slot_by_internal_player_id(self.parsed_dir, replay_id)
            for col in ("full_player_id", "player_id"):
                if col in self.participant_cols and all(row.get(col) is not None for row in participants):
                    mapped: dict[int, int] = {}
                    for row in participants:
                        slot = slot_by_internal.get(int(row[col]))
                        if slot in (1, 2):
                            mapped[slot] = int(row["profile_id"])
                    if len(mapped) == 2:
                        return mapped

        participant_ids = {int(row["profile_id"]) for row in participants}
        raw_order = _profile_order_from_raw(self.raw_dir, replay_id, participant_ids)
        if len(raw_order) == 2:
            return {1: raw_order[0], 2: raw_order[1]}

        if self.allow_profile_id_fallback:
            ordered = sorted(participant_ids)
            return {1: ordered[0], 2: ordered[1]}

        return None
