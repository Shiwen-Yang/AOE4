"""Live-match discovery for civ-choice prediction.

This module talks directly to the authenticated Relic/WorldsEdge Game API.
It intentionally does not implement Steam credential login; callers provide a
Relic `sessionID` obtained from an authenticated session.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable


GAME_API_BASE_URL = "https://aoe-api.worldsedgelink.com/game"


@dataclass(frozen=True)
class LivePlayer:
    match_id: int
    profile_id: int
    ping: int | None
    statgroup_id: int | None
    civilization_id: int | None
    team: int | None


@dataclass(frozen=True)
class LiveMatch:
    match_id: int
    leader_profile_id: int | None
    lobby_name: str | None
    match_name: str | None
    map_name: str | None
    options_b64: str | None
    slotinfo_b64: str | None
    game_mode_id: int | None
    is_ranked: bool | None
    started_at: datetime | None
    server_region: str | None
    players: tuple[LivePlayer, ...]

    @property
    def decoded_options(self) -> dict[str, Any]:
        return decode_relic_options(self.options_b64)

    @property
    def resolved_map_name(self) -> str | None:
        options_map = self.decoded_options.get("mapName")
        return options_map if isinstance(options_map, str) and options_map else self.map_name

    def contains_profile(self, profile_id: int) -> bool:
        return any(player.profile_id == profile_id for player in self.players)

    def opponents_of(self, profile_id: int) -> tuple[LivePlayer, ...]:
        teams = {player.team for player in self.players if player.profile_id == profile_id}
        if not teams:
            return tuple(player for player in self.players if player.profile_id != profile_id)
        return tuple(
            player
            for player in self.players
            if player.profile_id != profile_id and player.team not in teams
        )


def _as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _parse_started_at(value: Any) -> datetime | None:
    ts = _as_int(value)
    if ts is None or ts <= 0:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def decode_relic_options(options_b64: str | None) -> dict[str, Any]:
    """Decode Relic's base64/zlib JSON options payload when present."""
    if not options_b64:
        return {}
    try:
        raw = zlib.decompress(base64.b64decode(options_b64))
        decoded = json.loads(raw.decode("utf-8"))
    except (ValueError, zlib.error, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def parse_observable_advertisement(row: list[Any]) -> LiveMatch:
    """Parse one AoE4 `findObservableAdvertisements` row.

    The Game API returns positional arrays. The positions used here match the
    AoE4 examples in LibreMatch's documentation.
    """
    players_raw = row[14] if len(row) > 14 and isinstance(row[14], list) else []
    players: list[LivePlayer] = []
    for p in players_raw:
        if not isinstance(p, list):
            continue
        players.append(
            LivePlayer(
                match_id=_as_int(p[0]) or 0 if len(p) > 0 else 0,
                profile_id=_as_int(p[1]) or 0 if len(p) > 1 else 0,
                ping=_as_int(p[2]) if len(p) > 2 else None,
                statgroup_id=_as_int(p[3]) if len(p) > 3 else None,
                civilization_id=_as_int(p[4]) if len(p) > 4 else None,
                team=_as_int(p[5]) if len(p) > 5 else None,
            )
        )

    return LiveMatch(
        match_id=_as_int(row[0]) or 0 if len(row) > 0 else 0,
        leader_profile_id=_as_int(row[3]) if len(row) > 3 else None,
        lobby_name=_as_str(row[5]) if len(row) > 5 else None,
        match_name=_as_str(row[6]) if len(row) > 6 else None,
        map_name=_as_str(row[8]) if len(row) > 8 else None,
        options_b64=_as_str(row[9]) if len(row) > 9 else None,
        slotinfo_b64=_as_str(row[12]) if len(row) > 12 else None,
        game_mode_id=_as_int(row[13]) if len(row) > 13 else None,
        is_ranked=bool(row[15]) if len(row) > 15 and row[15] is not None else None,
        started_at=_parse_started_at(row[21]) if len(row) > 21 else None,
        server_region=_as_str(row[22]) if len(row) > 22 else None,
        players=tuple(players),
    )


def parse_find_observable_response(payload: Any) -> list[LiveMatch]:
    """Return live matches from a Game API response payload."""
    if not isinstance(payload, list) or len(payload) < 2:
        return []

    # AoE4 examples are `[0, [matches], [players]]`; some older examples are
    # `[matches, players]`. Accept both shapes.
    if isinstance(payload[1], list) and payload and payload[0] == 0:
        rows = payload[1]
    else:
        rows = payload[0]

    if not isinstance(rows, list):
        return []

    matches = []
    for row in rows:
        if isinstance(row, list) and row:
            matches.append(parse_observable_advertisement(row))
    return matches


class RelicGameApiClient:
    def __init__(
        self,
        session_id: str,
        *,
        base_url: str = GAME_API_BASE_URL,
        app_binary_checksum: int,
        data_checksum: int = 0,
        version_flags: int = 0,
        user_agent: str = "AOE4PredictionResearch/0.1",
    ) -> None:
        self.session_id = session_id
        self.base_url = base_url.rstrip("/")
        self.app_binary_checksum = app_binary_checksum
        self.data_checksum = data_checksum
        self.version_flags = version_flags
        self.user_agent = user_agent

    def find_observable_advertisements(
        self,
        *,
        profile_ids: Iterable[int] | None = None,
        count: int = 20,
        start: int = 0,
        timeout: int = 15,
    ) -> list[LiveMatch]:
        params: dict[str, Any] = {
            "appBinaryChecksum": self.app_binary_checksum,
            "callNum": 0,
            "connect_id": self.session_id,
            "count": count,
            "dataChecksum": self.data_checksum,
            "desc": 0,
            "lastCallTime": 0,
            "modDLLChecksum": 0,
            "modDLLFile": "INVALID",
            "modName": "INVALID",
            "modVersion": "INVALID",
            "observerGroupID": 0,
            "sessionID": self.session_id,
            "sortOrder": 0,
            "start": start,
            "versionFlags": self.version_flags,
        }
        if profile_ids:
            params["profile_ids"] = json.dumps([int(pid) for pid in profile_ids])

        url = f"{self.base_url}/advertisement/findObservableAdvertisements"
        request = urllib.request.Request(
            url + "?" + urllib.parse.urlencode(params),
            headers={"Accept": "application/json", "User-Agent": self.user_agent},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return parse_find_observable_response(payload)


def _format_match(match: LiveMatch, watched_profile_id: int) -> str:
    opponents = match.opponents_of(watched_profile_id)
    opp_text = ", ".join(
        f"{p.profile_id}/civ_id={p.civilization_id}/team={p.team}" for p in opponents
    ) or "unknown"
    started = match.started_at.isoformat() if match.started_at else "unknown"
    return (
        f"profile={watched_profile_id} match={match.match_id} "
        f"started={started} map={match.resolved_map_name} server={match.server_region} "
        f"opponents=[{opp_text}]"
    )


def monitor_profiles(args: argparse.Namespace) -> None:
    session_id = args.session_id or os.getenv("AOE4_RELIC_SESSION_ID")
    if not session_id:
        raise SystemExit("Provide --session-id or set AOE4_RELIC_SESSION_ID.")

    client = RelicGameApiClient(
        session_id=session_id,
        app_binary_checksum=args.app_binary_checksum,
        data_checksum=args.data_checksum,
        version_flags=args.version_flags,
    )
    profile_ids = [int(pid) for pid in args.profile_id]
    seen: dict[int, tuple[int, datetime | None]] = {}

    print(f"Monitoring profiles={profile_ids} interval={args.interval}s")
    while True:
        matches = client.find_observable_advertisements(
            profile_ids=profile_ids,
            count=args.count,
            timeout=args.timeout,
        )
        for profile_id in profile_ids:
            match = next((m for m in matches if m.contains_profile(profile_id)), None)
            if match is None:
                continue
            key = (match.match_id, match.started_at)
            if seen.get(profile_id) != key:
                print(datetime.now(timezone.utc).isoformat(), _format_match(match, profile_id), flush=True)
                seen[profile_id] = key
        time.sleep(args.interval)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Live AoE4 match discovery via Relic Game API")
    sub = parser.add_subparsers(dest="command", required=True)

    p_monitor = sub.add_parser("monitor", help="Poll observable matches for profile ids")
    p_monitor.add_argument("--profile-id", action="append", required=True, help="AoE4 profile id to watch")
    p_monitor.add_argument("--session-id", default=None, help="Relic Game API sessionID")
    p_monitor.add_argument("--app-binary-checksum", type=int, required=True, help="Current AoE4 app binary checksum/build")
    p_monitor.add_argument("--data-checksum", type=int, default=0, help="Current data checksum, if known")
    p_monitor.add_argument("--version-flags", type=int, default=0)
    p_monitor.add_argument("--count", type=int, default=20)
    p_monitor.add_argument("--interval", type=float, default=3.0)
    p_monitor.add_argument("--timeout", type=int, default=15)
    p_monitor.set_defaults(func=monitor_profiles)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
