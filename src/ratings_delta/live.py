"""Live conditional rating-delta inference for the backend.

Builds pre-match feature rows for two players directly from the
participants/games tables and scores both hypothetical outcomes (player A
wins, player B wins). Only information available before the match starts is
used — current rating/MMR, game counts, streak, recent form — never the
result, duration, or civ picks; the hypothetical winner enters only as the
conditioning variable.

Supports both delta models with their accepted rounding rules:
  - GBT (LightGBM booster, the deployment model): regular 0.5-threshold
    rounding (half away from zero)
  - P3 parametric: floor
"""
from __future__ import annotations

import datetime as _dt
from typing import Any

import numpy as np
import pandas as pd

from .model import predict_booster
from .parametric import P3Model

# Below this many games this season the rating system is in its placement
# phase and observed deltas are much more volatile (matches P2/P3 bucket 0).
PLACEMENT_GAMES = 10

# Window for streak / recent-form features (recent_wr_20 is the widest).
_RECENT_WINDOW = 20


def round_regular(x: float) -> int:
    """Regular 0.5-threshold rounding, half away from zero (np.round is half-to-even)."""
    return int(np.sign(x) * np.floor(np.abs(x) + 0.5))


def round_floor(x: float) -> int:
    return int(np.floor(x))


def get_current_season(conn, before_timestamp=None) -> int | None:
    """Latest season present in the DB (optionally as of before_timestamp)."""
    if before_timestamp is not None:
        row = conn.execute(
            "SELECT max(season) FROM games WHERE started_at < ?", [before_timestamp]
        ).fetchone()
    else:
        row = conn.execute("SELECT max(season) FROM games").fetchone()
    return row[0] if row else None


def get_current_patch(conn, before_timestamp=None) -> str | None:
    if before_timestamp is not None:
        row = conn.execute(
            "SELECT patch FROM games WHERE started_at < ? ORDER BY started_at DESC LIMIT 1",
            [before_timestamp],
        ).fetchone()
    else:
        row = conn.execute("SELECT patch FROM games ORDER BY started_at DESC LIMIT 1").fetchone()
    return row[0] if row else None


def get_live_player_state(
    conn,
    profile_id: int,
    season: int | None,
    before_timestamp=None,
) -> dict[str, Any]:
    """Current rating/MMR, game counts, and recent-form stats for one player.

    The dataset's `*_before` columns hold each player's rating going INTO a
    game, so the live equivalent is the post-match value of the most recent
    game: rating + rating_diff (and likewise for MMR).
    """
    ts_clause = "AND g.started_at < ?" if before_timestamp is not None else ""
    ts_params = [before_timestamp] if before_timestamp is not None else []

    rating_row = conn.execute(
        f"""
        SELECT p.rating + COALESCE(p.rating_diff, 0)
        FROM participants p
        JOIN games g ON p.game_id = g.game_id
        WHERE p.profile_id = ?
          AND g.kind IN ('rm_1v1', 'rm_solo')
          AND p.rating IS NOT NULL
          {ts_clause}
        ORDER BY g.started_at DESC
        LIMIT 1
        """,
        [profile_id] + ts_params,
    ).fetchone()

    mmr_row = conn.execute(
        f"""
        SELECT p.mmr + COALESCE(p.mmr_diff, 0)
        FROM participants p
        JOIN games g ON p.game_id = g.game_id
        WHERE p.profile_id = ?
          AND g.kind IN ('rm_1v1', 'rm_solo')
          AND p.mmr IS NOT NULL
          {ts_clause}
        ORDER BY g.started_at DESC
        LIMIT 1
        """,
        [profile_id] + ts_params,
    ).fetchone()

    lifetime_row = conn.execute(
        f"""
        SELECT count(*), max(g.started_at)
        FROM participants p
        JOIN games g ON p.game_id = g.game_id
        WHERE p.profile_id = ?
          AND g.kind IN ('rm_1v1', 'rm_solo')
          AND p.result IS NOT NULL
          {ts_clause}
        """,
        [profile_id] + ts_params,
    ).fetchone()
    games_lifetime = lifetime_row[0] if lifetime_row else 0
    last_game_at = lifetime_row[1] if lifetime_row else None

    if season is not None:
        games_row = conn.execute(
            f"""
            SELECT count(*)
            FROM participants p
            JOIN games g ON p.game_id = g.game_id
            WHERE p.profile_id = ?
              AND g.season = ?
              AND g.kind IN ('rm_1v1', 'rm_solo')
              AND p.result IS NOT NULL
              {ts_clause}
            """,
            [profile_id, season] + ts_params,
        ).fetchone()
        games_this_season = games_row[0] if games_row else 0
    else:
        games_this_season = 0

    # Most recent results (newest first) for streak and recent win rates.
    recent = [
        int(row[0])
        for row in conn.execute(
            f"""
            SELECT p.result::INT
            FROM participants p
            JOIN games g ON p.game_id = g.game_id
            WHERE p.profile_id = ?
              AND g.kind IN ('rm_1v1', 'rm_solo')
              AND p.result IS NOT NULL
              {ts_clause}
            ORDER BY g.started_at DESC
            LIMIT {_RECENT_WINDOW}
            """,
            [profile_id] + ts_params,
        ).fetchall()
    ]

    return {
        "profile_id": profile_id,
        "rating": rating_row[0] if rating_row else None,
        "mmr": mmr_row[0] if mmr_row else None,
        "games_lifetime": games_lifetime,
        "games_this_season": games_this_season,
        "last_game_at": last_game_at,
        "current_streak": _streak(recent),
        "recent_wr_10": float(np.mean(recent[:10])) if recent else None,
        "recent_wr_20": float(np.mean(recent[:20])) if recent else None,
    }


def _streak(recent: list[int]) -> int:
    """Signed run length of the most recent identical results (+wins / −losses)."""
    if not recent:
        return 0
    length = 1
    while length < len(recent) and recent[length] == recent[0]:
        length += 1
    return length if recent[0] == 1 else -length


def _days_since(last_game_at, reference: _dt.datetime) -> float:
    if last_game_at is None:
        return np.nan
    if hasattr(last_game_at, "to_pydatetime"):
        last_game_at = last_game_at.to_pydatetime()
    if last_game_at.tzinfo is not None:
        last_game_at = last_game_at.replace(tzinfo=None)
    return float((reference - last_game_at).days)


def _feature_row(
    player: dict,
    opponent: dict,
    result: int,
    reference: _dt.datetime,
    season: int | None,
    patch: str | None,
    map_name: str | None,
) -> dict[str, Any]:
    def _f(value):
        return float(value) if value is not None else np.nan

    rating_p, rating_o = _f(player["rating"]), _f(opponent["rating"])
    mmr_p, mmr_o = _f(player["mmr"]), _f(opponent["mmr"])

    return {
        "player_rating_before": rating_p,
        "opponent_rating_before": rating_o,
        "visible_rating_gap": rating_p - rating_o,
        "player_mmr_before": mmr_p,
        "opponent_mmr_before": mmr_o,
        "hidden_mmr_gap": mmr_p - mmr_o,
        "result": float(result),
        "games_lifetime_before": float(player["games_lifetime"]),
        "games_this_season_before": float(player["games_this_season"]),
        "opponent_games_this_season_before": float(opponent["games_this_season"]),
        "days_since_last_game": _days_since(player["last_game_at"], reference),
        "current_streak": float(player["current_streak"]),
        "recent_wr_10": _f(player["recent_wr_10"]),
        "recent_wr_20": _f(player["recent_wr_20"]),
        "missing_player_rating": float(player["rating"] is None),
        "missing_opponent_rating": float(opponent["rating"] is None),
        "missing_player_mmr": float(player["mmr"] is None),
        "missing_opponent_mmr": float(opponent["mmr"] is None),
        "season": season,
        "patch": patch,
        "map": map_name,
    }


def predict_conditional_deltas(
    model,
    conn,
    player_a_id: int,
    player_b_id: int,
    before_timestamp=None,
    map_name: str | None = None,
) -> dict[str, Any]:
    """Predict each player's rating delta under both match outcomes.

    Returns per-player dicts with current rating/MMR, season game count, and
    delta_if_win / delta_if_loss, plus data-quality warnings. Deltas are
    rounded to whole points with each model's accepted rule: regular
    0.5-threshold rounding for the GBT, floor for the P3 parametric model.
    Deltas are None when a player has no rating or MMR history at all.
    """
    season = get_current_season(conn, before_timestamp)
    patch = get_current_patch(conn, before_timestamp)
    state_a = get_live_player_state(conn, player_a_id, season, before_timestamp)
    state_b = get_live_player_state(conn, player_b_id, season, before_timestamp)
    return _score_conditional_deltas(
        model, state_a, state_b, season, patch, map_name, reference=before_timestamp
    )


def _normalize_reference(reference) -> _dt.datetime:
    reference = reference or _dt.datetime.utcnow()
    if hasattr(reference, "to_pydatetime"):
        reference = reference.to_pydatetime()
    if reference.tzinfo is not None:
        reference = reference.replace(tzinfo=None)
    return reference


def _score_conditional_deltas(
    model,
    state_a: dict[str, Any],
    state_b: dict[str, Any],
    season: int | None,
    patch: str | None,
    map_name: str | None,
    reference=None,
    extra_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Score both hypothetical outcomes for two already-built player states.

    Shared by the DB path (`predict_conditional_deltas`) and the live aoe4world
    path (`predict_conditional_deltas_from_games`) so both produce identical
    payloads from identical state dicts.
    """
    reference = _normalize_reference(reference)

    frame = pd.DataFrame(
        [
            _feature_row(state_a, state_b, 1, reference, season, patch, map_name),
            _feature_row(state_a, state_b, 0, reference, season, patch, map_name),
            _feature_row(state_b, state_a, 1, reference, season, patch, map_name),
            _feature_row(state_b, state_a, 0, reference, season, patch, map_name),
        ]
    )

    if isinstance(model, P3Model):
        preds = np.asarray(model.predict(frame), dtype=float)
        rounder = round_floor
    else:  # LightGBM booster (deployment model)
        preds = np.asarray(predict_booster(model, frame), dtype=float)
        rounder = round_regular

    # A player with no rating signal at all cannot be scored; the GBT would
    # happily extrapolate from all-NaN inputs, so null the deltas explicitly.
    no_signal = [
        state["rating"] is None and state["mmr"] is None for state in (state_a, state_b)
    ]

    def _val(idx: int, player_pos: int) -> int | None:
        if no_signal[player_pos] or np.isnan(preds[idx]):
            return None
        return rounder(preds[idx])

    warnings: list[str] = list(extra_warnings or [])
    for label, state in (("Player A", state_a), ("Player B", state_b)):
        if state["rating"] is None and state["mmr"] is None:
            warnings.append(
                f"{label} has no rating or MMR history; rating delta cannot be estimated."
            )
        elif state["mmr"] is None:
            warnings.append(
                f"{label} has no MMR history; delta uses visible-rating fallback."
            )
        if state["games_this_season"] < PLACEMENT_GAMES:
            warnings.append(
                f"{label} has played {state['games_this_season']} games this season "
                f"(< {PLACEMENT_GAMES}); placement-phase deltas are more volatile."
            )

    return {
        "season": season,
        "player_a": {**state_a, "delta_if_win": _val(0, 0), "delta_if_loss": _val(1, 0)},
        "player_b": {**state_b, "delta_if_win": _val(2, 1), "delta_if_loss": _val(3, 1)},
        "warnings": warnings,
    }


def player_state_from_games(
    profile_id: int,
    games: list[dict],
    season: int | None,
    now: _dt.datetime | None = None,
) -> dict[str, Any]:
    """Build the same player-state dict as `get_live_player_state`, but from a
    player's aoe4world recent-games list (most-recent-first) instead of DuckDB.

    Game counts are limited to what the recent-games page returns, so a
    high-volume player's `games_lifetime`/`games_this_season` are floors, not
    true totals — the inherent cost of DB-free serving. Rating/MMR are the
    latest non-null values (the player's current skill), matching how the live
    outcome features define skill.
    """
    now = _normalize_reference(now)
    rating = next((g.get("rating") for g in games if g.get("rating") is not None), None)
    mmr = next((g.get("mmr") for g in games if g.get("mmr") is not None), None)
    results = [int(g.get("result") or 0) for g in games if g.get("result") is not None]
    games_this_season = sum(1 for g in games if g.get("season") == season)
    return {
        "profile_id": profile_id,
        "rating": rating,
        "mmr": mmr,
        "games_lifetime": len(games),
        "games_this_season": games_this_season,
        "last_game_at": games[0].get("started_at") if games else None,
        "current_streak": _streak(results),
        "recent_wr_10": float(np.mean(results[:10])) if results else None,
        "recent_wr_20": float(np.mean(results[:20])) if results else None,
    }


def predict_conditional_deltas_from_games(
    model,
    games_a: list[dict],
    games_b: list[dict],
    player_a_id: int,
    player_b_id: int,
    season: int | None,
    patch: str | None,
    map_name: str | None = None,
    now: _dt.datetime | None = None,
) -> dict[str, Any]:
    """DB-free conditional rating deltas from two players' recent-games lists.

    The live counterpart to `predict_conditional_deltas`: it reuses the exact
    aoe4world data the outcome prediction already fetched, so the rating-point
    estimate rides off the same API call with no database and no extra request.
    """
    now = _normalize_reference(now)
    state_a = player_state_from_games(player_a_id, games_a, season, now)
    state_b = player_state_from_games(player_b_id, games_b, season, now)
    extra = [
        "Rating deltas use aoe4world recent-games only; season game counts are "
        "page-limited, so deltas for high-volume players are approximate."
    ]
    return _score_conditional_deltas(
        model, state_a, state_b, season, patch, map_name, reference=now, extra_warnings=extra
    )
