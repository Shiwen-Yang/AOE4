"""
Live aoe4world client for DB-free outcome inference.

Two reads back a prediction:
  - fetch_recent_games(profile_id): one recent-games page per player (×2, concurrent).
  - fetch_matchup_priors(): one global civ-matchup snapshot, cached for hours.

HTTP uses stdlib urllib (no new deps), mirroring
src/realtime_outcome_prediction/outcomes.py. The JSON fetcher is injectable so
tests can stub responses without network access.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger("aoe4.backend.aoe4world")

DEFAULT_BASE_URL = "https://aoe4world.com"
DEFAULT_USER_AGENT = "AOE4Predict/0.1 (outcome-inference)"

#: Returns parsed JSON for a URL; raises urllib.error.HTTPError/URLError on failure.
JsonFetcher = Callable[[str], Any]


class Aoe4WorldError(RuntimeError):
    """Base class for aoe4world client failures."""


class Aoe4WorldUnavailable(Aoe4WorldError):
    """Endpoint failed after retries (timeout, 5xx, or persistent 429)."""


def _result_with_label(future, label: str):
    """Resolve a future, prefixing the player label while preserving the error
    subtype (Unavailable → 503, plain Aoe4WorldError/parse → 502)."""
    try:
        return future.result()
    except Aoe4WorldUnavailable as exc:
        raise Aoe4WorldUnavailable(f"{label}: {exc}") from exc
    except Aoe4WorldError as exc:
        raise Aoe4WorldError(f"{label}: {exc}") from exc


def _parse_ts(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        s = str(raw).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _result_to_int(raw: Any) -> int | None:
    if raw == "win":
        return 1
    if raw == "loss":
        return 0
    return None


def _iter_players(game: dict) -> list[dict]:
    """Flatten teams→players (handles nested {'player': {...}} and flat dicts)."""
    out: list[dict] = []
    for team in game.get("teams") or []:
        for entry in team or []:
            player = entry.get("player") if isinstance(entry, dict) and "player" in entry else entry
            if isinstance(player, dict) and player.get("profile_id") is not None:
                out.append(player)
    return out


def _default_fetch(url: str, timeout: float, user_agent: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class _TTLCache:
    def __init__(self, ttl_seconds: float):
        self.ttl = ttl_seconds
        self._store: dict[Any, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: Any) -> Any | None:
        with self._lock:
            hit = self._store.get(key)
            if hit and (time.monotonic() - hit[0]) < self.ttl:
                return hit[1]
        return None

    def put(self, key: Any, value: Any) -> None:
        with self._lock:
            self._store[key] = (time.monotonic(), value)


class Aoe4WorldClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        mode: str = "rm_solo",
        timeout: float = 8.0,
        retries: int = 2,
        user_agent: str = DEFAULT_USER_AGENT,
        games_ttl: float = 45.0,
        matchups_ttl: float = 6 * 3600.0,
        json_fetcher: JsonFetcher | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.mode = mode
        self.timeout = timeout
        self.retries = retries
        self.user_agent = user_agent
        self._fetch = json_fetcher or (lambda url: _default_fetch(url, self.timeout, self.user_agent))
        self._games_cache = _TTLCache(games_ttl)
        self._matchups_cache = _TTLCache(matchups_ttl)
        self._matchups_stale: dict[tuple, tuple[int, int]] | None = None

    # ── HTTP with retries/backoff ────────────────────────────────────────────
    def _get(self, url: str) -> Any:
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return self._fetch(url)
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    raise  # caller decides (new player vs hard error)
                if exc.code not in (429, 500, 502, 503, 504):
                    raise Aoe4WorldUnavailable(f"{url} -> HTTP {exc.code}") from exc
                last = exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
            if attempt < self.retries:
                time.sleep(0.5 * (2 ** attempt))
        raise Aoe4WorldUnavailable(f"{url} failed after {self.retries + 1} attempts: {last}")

    # ── recent games ─────────────────────────────────────────────────────────
    def fetch_recent_games(self, profile_id: int) -> list[dict]:
        """One recent-games page, normalized to inference_api.Game dicts (most-recent-first).
        Returns [] for an unknown/new player (404 or empty)."""
        cached = self._games_cache.get(profile_id)
        if cached is not None:
            return cached
        url = f"{self.base_url}/api/v0/players/{int(profile_id)}/games?mode={self.mode}&page=1"
        try:
            payload = self._get(url)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                self._games_cache.put(profile_id, [])
                return []
            raise Aoe4WorldUnavailable(f"{url} -> HTTP {exc.code}") from exc
        games = self._parse_games(payload, profile_id)
        self._games_cache.put(profile_id, games)
        return games

    def _parse_games(self, payload: Any, profile_id: int) -> list[dict]:
        raw_games = payload.get("games") if isinstance(payload, dict) else payload
        if not isinstance(raw_games, list):
            raise Aoe4WorldError("unexpected /games payload shape")
        out: list[dict] = []
        for g in raw_games:
            if not isinstance(g, dict):
                continue
            players = _iter_players(g)
            me = next((p for p in players if int(p.get("profile_id", -1)) == int(profile_id)), None)
            if me is None:
                continue
            opp = next((p for p in players if int(p.get("profile_id", -1)) != int(profile_id)), None)
            out.append({
                "game_id": g.get("game_id"),
                "started_at": _parse_ts(g.get("started_at")),
                "duration": g.get("duration"),
                "map": g.get("map"),
                "season": g.get("season"),
                "patch": g.get("patch"),
                "civ": me.get("civilization"),
                "result": _result_to_int(me.get("result")),
                "rating": me.get("rating"),
                "mmr": me.get("mmr"),
                "opponent_profile_id": opp.get("profile_id") if opp else None,
                "opponent_civ": opp.get("civilization") if opp else None,
            })
        # Most-recent-first; drop games without a usable result or timestamp.
        out = [g for g in out if g["result"] is not None and g["started_at"] is not None]
        out.sort(key=lambda g: g["started_at"], reverse=True)
        return out

    def fetch_both(self, profile_a: int, profile_b: int) -> tuple[list[dict], list[dict]]:
        """Fetch both players' recent games concurrently. Raises Aoe4WorldUnavailable
        (naming the failed player) if either side errors."""
        with ThreadPoolExecutor(max_workers=2) as pool:
            fa = pool.submit(self.fetch_recent_games, profile_a)
            fb = pool.submit(self.fetch_recent_games, profile_b)
            ga = _result_with_label(fa, f"player_a {profile_a}")
            gb = _result_with_label(fb, f"player_b {profile_b}")
        return ga, gb

    # ── civ-matchup priors (global snapshot, cached) ─────────────────────────
    def fetch_matchup_priors(self, patch: Any = None) -> dict[tuple, tuple[int, int]]:
        """(civ, other_civ) -> (games_count, win_count). Served from cache; on a
        fresh fetch failure, falls back to the last good snapshot (or {})."""
        key = ("matchups", patch)
        cached = self._matchups_cache.get(key)
        if cached is not None:
            return cached
        url = f"{self.base_url}/api/v0/stats/{self.mode}/matchups"
        if patch is not None:
            url += "?" + urllib.parse.urlencode({"patch": patch})
        try:
            payload = self._get(url)
            snapshot = self._parse_matchups(payload)
            self._matchups_cache.put(key, snapshot)
            self._matchups_stale = snapshot
            return snapshot
        except (Aoe4WorldError, urllib.error.HTTPError) as exc:
            logger.warning("matchups fetch failed (%s); using stale snapshot", exc)
            return self._matchups_stale or {}

    @staticmethod
    def _parse_matchups(payload: Any) -> dict[tuple, tuple[int, int]]:
        records = payload
        if isinstance(payload, dict):
            for key in ("data", "matchups", "stats"):
                if isinstance(payload.get(key), list):
                    records = payload[key]
                    break
        if not isinstance(records, list):
            raise Aoe4WorldError("unexpected /matchups payload shape")
        out: dict[tuple, tuple[int, int]] = {}
        for r in records:
            if not isinstance(r, dict):
                continue
            civ = r.get("civilization")
            other = r.get("other_civilization")
            if civ is None or other is None:
                continue
            games = int(r.get("games_count") or 0)
            wins = r.get("win_count")
            if wins is None and r.get("win_rate") is not None:
                wins = round(float(r["win_rate"]) / 100.0 * games)
            out[(civ, other)] = (games, int(wins or 0))
        return out

    def matchup_lookup(self, snapshot: dict[tuple, tuple[int, int]]):
        """Return a build_recent_only_features-compatible (civ_a, civ_b) -> (games, wins).

        Mirror matchups (civ_a == civ_b) are forced to a symmetric 50% win rate:
        the aoe4world matchups endpoint reports win_count == games_count (100%) for
        mirrors, which is degenerate. By symmetry the true value is 0.5.
        """
        def lookup(civ_a: str | None, civ_b: str | None) -> tuple[int, int]:
            if civ_a is None or civ_b is None:
                return (0, 0)
            games, wins = snapshot.get((civ_a, civ_b), (0, 0))
            if civ_a == civ_b:
                return (games, games // 2)
            return (games, wins)
        return lookup
