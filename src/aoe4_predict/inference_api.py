"""
Build the API30 ``recent_only`` feature vector from each player's recent games
(as returned by the aoe4world recent-games endpoint) instead of from DuckDB.

This is the live, DB-free counterpart to ``get_inference_features`` +
``get_extended_inference_features``. It reproduces, over a Python list of a
player's most-recent games, the exact cap-30 semantics the ``recent_only`` model
was trained on:

  - base lifetime/season/civ/map counts  ← build_recent_only_base_overrides
  - P1 civ-recency, P4 duration-profile  ← build_api_cap_p1_p3_p4_p5_overrides
  - P3 recent form (last 5/10/20 games, already within the window)
  - P5 head-to-head (meetings visible in *both* players' capped windows)

MMR/rating are taken from the most recent game (not capped — the API returns
them per game). Civ-matchup priors come from an injected lookup (an aoe4world
matchups snapshot in production; the DuckDB table in the parity test).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable, Sequence

from .config import GLOBAL_WR_PRIOR, NEW_PLAYER_THRESHOLD, PRIOR_STRENGTH
from .features import _apply_cold_start_skill_priors

#: Recent-window cap — one aoe4world page worth of prior games.
VISIBLE_MATCH_CAP = 30
H2H_PRIOR = 5

#: A normalized recent game. ``result`` is 1 (win) / 0 (loss) / None.
Game = dict[str, Any]
#: civ_a, civ_b → (prior_games, prior_wins-of-civ_a)
MatchupLookup = Callable[[str | None, str | None], tuple[int, int]]


def _smooth(wins: float, games: float, p: int = PRIOR_STRENGTH, g: float = GLOBAL_WR_PRIOR) -> float:
    return (wins + p * g) / (games + p)


def _result(game: Game) -> int:
    return int(game.get("result") or 0)


def _days_since(now: datetime, ts: datetime | None) -> int | None:
    # Calendar-day difference, matching DuckDB DATEDIFF('day', ts, now) used in
    # training (player_stats.days_since_last_game / capped days_since_civ).
    if ts is None:
        return None
    if ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)
    return (now.date() - ts.date()).days


def _within(game: Game, now: datetime, days: int) -> bool:
    # Matches training's `started_at >= now - INTERVAL 'N days'` (inclusive).
    ts = game.get("started_at")
    if ts is None:
        return False
    if ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)
    return ts >= now - timedelta(days=days)


def _side_raw(
    games: Sequence[Game],
    civ: str | None,
    map_name: str | None,
    season: Any,
    now: datetime,
) -> dict[str, Any]:
    """Raw cap-30 counts for one player, mirroring the SQL override builders."""
    visible = list(games[:VISIBLE_MATCH_CAP])
    n = len(visible)

    # MMR/rating are NOT capped: latest non-null over the full returned history.
    last_mmr = next((g.get("mmr") for g in games if g.get("mmr") is not None), None)
    last_rating = next((g.get("rating") for g in games if g.get("rating") is not None), None)
    last_game_at = games[0].get("started_at") if games else None

    civ_games = [g for g in visible if g.get("civ") == civ] if civ is not None else []
    map_games = [g for g in visible if g.get("map") == map_name] if map_name is not None else []
    season_games = [g for g in visible if g.get("season") == season]

    durs = [g["duration"] for g in visible if g.get("duration") is not None]
    civ_durs = [g["duration"] for g in civ_games if g.get("duration") is not None]
    last20 = visible[:20]
    dur20 = [g["duration"] for g in last20 if g.get("duration") is not None]

    short = [g for g in visible if g.get("duration") is not None and g["duration"] <= 900]
    long_ = [g for g in visible if g.get("duration") is not None and g["duration"] > 1800]

    def civ_window(days: int) -> tuple[int, int]:
        gs = [g for g in civ_games if _within(g, now, days)]
        return len(gs), sum(_result(g) for g in gs)

    cg7, cw7 = civ_window(7)
    cg30, cw30 = civ_window(30)
    cg60, cw60 = civ_window(60)
    act30 = sum(1 for g in visible if _within(g, now, 30))
    act60 = sum(1 for g in visible if _within(g, now, 60))

    last_civ_at = max((g["started_at"] for g in civ_games if g.get("started_at")), default=None)

    # P3 recent form (most recent 5/10/20 games — already inside the window).
    results = [_result(g) for g in games]
    recent = {}
    for k in (5, 10, 20):
        recent[f"recent_n_{k}"] = min(k, len(results))
        recent[f"recent_w_{k}"] = int(sum(results[:k]))

    return {
        "visible_games": n,
        "last_mmr": last_mmr,
        "last_rating": last_rating,
        "last_game_at": last_game_at,
        # base cap-30 counts
        "games_lifetime": n,
        "wins_lifetime": sum(_result(g) for g in visible),
        "games_season": len(season_games),
        "wins_season": sum(_result(g) for g in season_games),
        "civ_games": len(civ_games),
        "civ_wins": sum(_result(g) for g in civ_games),
        "map_games": len(map_games),
        "map_wins": sum(_result(g) for g in map_games),
        # P4 duration profile
        "avg_dur_life": (sum(durs) / len(durs)) if durs else None,
        "civ_avg_dur": (sum(civ_durs) / len(civ_durs)) if civ_durs else None,
        "avg_dur_20": (sum(dur20) / len(dur20)) if dur20 else None,
        "short_games": len(short),
        "short_wins": sum(_result(g) for g in short),
        "long_games": len(long_),
        "long_wins": sum(_result(g) for g in long_),
        # P1 civ recency windows
        "civ_games_7d": cg7, "civ_wins_7d": cw7,
        "civ_games_30d": cg30, "civ_wins_30d": cw30,
        "civ_games_60d": cg60, "civ_wins_60d": cw60,
        "act_games_30d": act30, "act_games_60d": act60,
        "days_since_civ": _days_since(now, last_civ_at),
        **recent,
    }


def build_recent_only_features(
    games_a: Sequence[Game],
    games_b: Sequence[Game],
    profile_a: int,
    profile_b: int,
    civ_a: str | None,
    civ_b: str | None,
    map_name: str | None,
    matchup_lookup: MatchupLookup,
    now: datetime | None = None,
    season: Any = None,
    patch: Any = None,
) -> dict[str, Any]:
    """
    Return the ordered feature dict the API30 ``recent_only`` model consumes,
    computed from two players' recent-games lists (most-recent-first).
    """
    now = now or datetime.utcnow()
    games_a = list(games_a)
    games_b = list(games_b)

    # Current match season/patch: the most recent game across both players.
    if season is None or patch is None:
        latest = max(
            (g for g in (games_a[:1] + games_b[:1]) if g.get("started_at")),
            key=lambda g: g["started_at"],
            default=None,
        )
        if latest is not None:
            season = season if season is not None else latest.get("season")
            patch = patch if patch is not None else latest.get("patch")

    ra = _side_raw(games_a, civ_a, map_name, season, now)
    rb = _side_raw(games_b, civ_b, map_name, season, now)

    prior_games, prior_wins = matchup_lookup(civ_a, civ_b)

    feat: dict[str, Any] = {
        "season": season, "patch": patch, "map": map_name,
        "civ_a": civ_a, "civ_b": civ_b,
        "mmr_a": ra["last_mmr"], "rating_a": ra["last_rating"],
        "mmr_b": rb["last_mmr"], "rating_b": rb["last_rating"],
        "games_lifetime_a": ra["games_lifetime"], "wins_lifetime_a": ra["wins_lifetime"],
        "games_lifetime_b": rb["games_lifetime"], "wins_lifetime_b": rb["wins_lifetime"],
        "games_season_a": ra["games_season"], "wins_season_a": ra["wins_season"],
        "games_season_b": rb["games_season"], "wins_season_b": rb["wins_season"],
        "days_since_a": _days_since(now, ra["last_game_at"]),
        "days_since_b": _days_since(now, rb["last_game_at"]),
        "civ_games_a": ra["civ_games"], "civ_wins_a": ra["civ_wins"],
        "civ_games_b": rb["civ_games"], "civ_wins_b": rb["civ_wins"],
        "map_games_a": ra["map_games"], "map_wins_a": ra["map_wins"],
        "map_games_b": rb["map_games"], "map_wins_b": rb["map_wins"],
        "prior_matchup_games": prior_games, "prior_matchup_wins": prior_wins,
    }

    # ── base derived (mirror features.get_inference_features) ──────────────────
    feat["overall_wr_a"] = _smooth(feat["wins_lifetime_a"], feat["games_lifetime_a"])
    feat["overall_wr_b"] = _smooth(feat["wins_lifetime_b"], feat["games_lifetime_b"])
    feat["season_wr_a"] = _smooth(feat["wins_season_a"], feat["games_season_a"])
    feat["season_wr_b"] = _smooth(feat["wins_season_b"], feat["games_season_b"])
    feat["civ_wr_a"] = _smooth(feat["civ_wins_a"], feat["civ_games_a"])
    feat["civ_wr_b"] = _smooth(feat["civ_wins_b"], feat["civ_games_b"])
    feat["map_wr_a"] = _smooth(feat["map_wins_a"], feat["map_games_a"])
    feat["map_wr_b"] = _smooth(feat["map_wins_b"], feat["map_games_b"])
    feat["prior_matchup_wr_a"] = _smooth(feat["prior_matchup_wins"], feat["prior_matchup_games"])

    skill_a = feat["mmr_a"] if feat["mmr_a"] is not None else feat["rating_a"]
    skill_b = feat["mmr_b"] if feat["mmr_b"] is not None else feat["rating_b"]
    feat["skill_a"], feat["skill_b"] = skill_a, skill_b
    feat["missing_mmr_a"] = int(feat["mmr_a"] is None)
    feat["missing_mmr_b"] = int(feat["mmr_b"] is None)
    feat["missing_rating_a"] = int(feat["rating_a"] is None)
    feat["missing_rating_b"] = int(feat["rating_b"] is None)
    feat["missing_skill_a"] = int(skill_a is None)
    feat["missing_skill_b"] = int(skill_b is None)
    feat["mmr_diff"] = (feat["mmr_a"] or 0) - (feat["mmr_b"] or 0)
    feat["rating_diff"] = (feat["rating_a"] or 0) - (feat["rating_b"] or 0)
    feat["skill_diff"] = (skill_a or 0) - (skill_b or 0)
    feat["games_diff"] = feat["games_lifetime_a"] - feat["games_lifetime_b"]
    feat["wr_diff"] = feat["overall_wr_a"] - feat["overall_wr_b"]
    feat["is_new_player_a"] = int(feat["games_lifetime_a"] < NEW_PLAYER_THRESHOLD)
    feat["is_new_player_b"] = int(feat["games_lifetime_b"] < NEW_PLAYER_THRESHOLD)
    feat["civs_known"] = int(civ_a is not None and civ_b is not None)
    feat["map_known"] = int(map_name is not None)
    feat["full_context_known"] = int(feat["civs_known"] and feat["map_known"])
    feat = _apply_cold_start_skill_priors(feat)

    # ── P1 + P4 raw + derived (mirror apply_api_cap_p1_p3_p4_p5_overrides) ─────
    for side, r in (("a", ra), ("b", rb)):
        for col in (
            "civ_games_7d", "civ_wins_7d", "civ_games_30d", "civ_wins_30d",
            "civ_games_60d", "civ_wins_60d", "days_since_civ",
            "avg_dur_life", "avg_dur_20", "civ_avg_dur",
            "short_games", "short_wins", "long_games", "long_wins",
        ):
            feat[f"{col}_{side}"] = r[col]
        feat[f"civ_wr_7d_{side}"] = _smooth(r["civ_wins_7d"], r["civ_games_7d"])
        feat[f"civ_wr_30d_{side}"] = _smooth(r["civ_wins_30d"], r["civ_games_30d"])
        feat[f"civ_wr_60d_{side}"] = _smooth(r["civ_wins_60d"], r["civ_games_60d"])
        feat[f"civ_frac_30d_{side}"] = r["civ_games_30d"] / max(r["act_games_30d"], 1)
        feat[f"civ_frac_60d_{side}"] = r["civ_games_60d"] / max(r["act_games_60d"], 1)
        feat[f"short_wr_{side}"] = _smooth(r["short_wins"], r["short_games"])
        feat[f"long_wr_{side}"] = _smooth(r["long_wins"], r["long_games"])
        visible = max(r["visible_games"], 1)
        feat[f"short_share_{side}"] = r["short_games"] / visible
        feat[f"long_share_{side}"] = r["long_games"] / visible

    feat["civ_wr_30d_diff"] = feat["civ_wr_30d_a"] - feat["civ_wr_30d_b"]
    feat["civ_wr_60d_diff"] = feat["civ_wr_60d_a"] - feat["civ_wr_60d_b"]
    feat["civ_frac_30d_diff"] = feat["civ_frac_30d_a"] - feat["civ_frac_30d_b"]
    feat["civ_frac_60d_diff"] = feat["civ_frac_60d_a"] - feat["civ_frac_60d_b"]
    feat["avg_dur_life_diff"] = (feat["avg_dur_life_a"] or 0) - (feat["avg_dur_life_b"] or 0)
    feat["avg_dur_20_diff"] = (feat["avg_dur_20_a"] or 0) - (feat["avg_dur_20_b"] or 0)
    feat["civ_avg_dur_diff"] = (feat["civ_avg_dur_a"] or 0) - (feat["civ_avg_dur_b"] or 0)
    feat["short_wr_diff"] = feat["short_wr_a"] - feat["short_wr_b"]
    feat["long_wr_diff"] = feat["long_wr_a"] - feat["long_wr_b"]
    feat["short_share_diff"] = feat["short_share_a"] - feat["short_share_b"]
    feat["long_share_diff"] = feat["long_share_a"] - feat["long_share_b"]

    # ── P3 recent form raw + derived ──────────────────────────────────────────
    for side, r in (("a", ra), ("b", rb)):
        for k in (5, 10, 20):
            feat[f"recent_w_{k}_{side}"] = r[f"recent_w_{k}"]
            feat[f"recent_wr_{k}_{side}"] = _smooth(r[f"recent_w_{k}"], r[f"recent_n_{k}"])
    for k in (5, 10, 20):
        feat[f"recent_wr_{k}_diff"] = feat[f"recent_wr_{k}_a"] - feat[f"recent_wr_{k}_b"]

    # ── P5 head-to-head (meetings visible in BOTH capped windows) ─────────────
    visible_b_ids = {g.get("game_id") for g in games_b[:VISIBLE_MATCH_CAP]}
    h2h = [
        g for g in games_a[:VISIBLE_MATCH_CAP]
        if g.get("opponent_profile_id") == profile_b and g.get("game_id") in visible_b_ids
    ]
    feat["h2h_games"] = len(h2h)
    feat["h2h_wins_a"] = sum(_result(g) for g in h2h)
    feat["h2h_wr_a"] = (feat["h2h_wins_a"] + H2H_PRIOR * 0.5) / (feat["h2h_games"] + H2H_PRIOR)

    return feat
