"""Backend outcome endpoint over the live (api) feature source, with a stubbed
aoe4world fetcher — no network. Covers the happy path and the main failure modes.
"""
from datetime import datetime, timedelta
from pathlib import Path
import re
import sys
import urllib.error

import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from aoe4_predict.model import load_model
from backend.app.resources import (
    API30_MODEL_META_PATH,
    API30_MODEL_PATH,
    DELTA_PARAMETRIC_PATH,
    AppResources,
    Settings,
    load_parametric_model,
    load_trained_categories,
)
from backend.app.services.aoe4world_client import Aoe4WorldClient
from backend.app.services.outcome import predict_outcome
from backend.app.validation import prepare_outcome_request
from backend.app.schemas import OutcomePredictionRequest
from datetime import timezone


def _games_for(pid, opp_pid, n=35, civ="english"):
    base = datetime(2026, 5, 1)
    games = []
    for i in range(n):
        ts = (base - timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        games.append({
            "game_id": pid * 100000 + i,
            "started_at": ts,
            "duration": [600, 1200, 2400][i % 3],
            "map": "Dry Arabia", "season": 12, "patch": "10.1",
            "teams": [
                [{"player": {"profile_id": pid, "result": "win" if i % 2 == 0 else "loss",
                             "civilization": civ, "rating": 1500, "mmr": 1500}}],
                [{"player": {"profile_id": opp_pid, "result": "loss" if i % 2 == 0 else "win",
                             "civilization": "french", "rating": 1490, "mmr": 1490}}],
            ],
        })
    return games


MATCHUPS = [
    {"civilization": "english", "other_civilization": "french",
     "games_count": 1000, "win_count": 520, "win_rate": 52.0},
]


def _fetcher(games_by_pid, fail_pids=None, malformed_pid=None):
    fail_pids = fail_pids or set()

    def fetch(url):
        if "/matchups" in url:
            return {"data": MATCHUPS}
        pid = int(re.search(r"/players/(\d+)/games", url).group(1))
        if pid in fail_pids:
            raise urllib.error.HTTPError(url, 503, "boom", {}, None)
        if pid == malformed_pid:
            return {"unexpected": "shape"}
        if pid not in games_by_pid:
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)
        return {"games": games_by_pid[pid]}

    return fetch


def _resources(fetch) -> AppResources:
    settings = Settings.from_env()  # AOE4_FEATURE_SOURCE defaults to "api"
    assert settings.feature_source == "api"
    model, meta = load_model(API30_MODEL_PATH, API30_MODEL_META_PATH)
    client = Aoe4WorldClient(retries=0, json_fetcher=fetch)
    return AppResources(
        settings=settings, model=model, meta=meta,
        trained_categories=load_trained_categories(API30_MODEL_PATH, meta),
        db_metadata={"db_civs": [], "db_maps": [], "latest_patch": None, "latest_season": None},
        loaded_at=datetime.now(timezone.utc), aoe4world_client=client,
    )


def _prepare(res, a=100, b=200, civ_a="english", civ_b="french", map_name="Dry Arabia"):
    req = OutcomePredictionRequest(
        player_a_id=a, player_b_id=b, civ_a=civ_a, civ_b=civ_b, map_name=map_name,
    )
    return prepare_outcome_request(req, res)


def test_live_happy_path_contract():
    games = {100: _games_for(100, 200), 200: _games_for(200, 100, civ="french")}
    res = _resources(_fetcher(games))
    out = predict_outcome(_prepare(res), res)
    assert 0.0 <= out.prediction.win_prob_a <= 1.0
    assert out.prediction.win_prob_a + out.prediction.win_prob_b == pytest.approx(1.0, abs=1e-3)
    assert out.data_quality.context_level == "full_context"


def test_live_unknown_player_is_cold_start_not_error():
    # Player 200 returns 404 → treated as new player with cold-start priors.
    games = {100: _games_for(100, 200)}
    res = _resources(_fetcher(games))
    out = predict_outcome(_prepare(res), res)
    assert 0.0 <= out.prediction.win_prob_a <= 1.0
    assert any("Player B" in w for w in out.data_quality.warnings)


def test_live_one_player_unavailable_returns_503():
    games = {100: _games_for(100, 200), 200: _games_for(200, 100)}
    res = _resources(_fetcher(games, fail_pids={200}))
    with pytest.raises(HTTPException) as ei:
        predict_outcome(_prepare(res), res)
    assert ei.value.status_code == 503
    assert "player_b" in str(ei.value.detail)


def test_matchup_lookup_neutralizes_mirror_matchups():
    client = Aoe4WorldClient(retries=0, json_fetcher=lambda url: {"data": []})
    # Snapshot reports a degenerate 100% win rate for the mirror matchup.
    snap = {("english", "english"): (578, 578), ("english", "french"): (1000, 520)}
    lookup = client.matchup_lookup(snap)
    g, w = lookup("english", "english")
    assert (g, w) == (578, 289)  # forced to symmetric 50%
    assert lookup("english", "french") == (1000, 520)  # non-mirror untouched


def test_live_malformed_payload_returns_502():
    games = {200: _games_for(200, 100)}
    res = _resources(_fetcher(games, malformed_pid=100))
    with pytest.raises(HTTPException) as ei:
        predict_outcome(_prepare(res), res)
    assert ei.value.status_code == 502


def test_live_outcome_embeds_rating_delta_from_same_fetch():
    games = {100: _games_for(100, 200), 200: _games_for(200, 100, civ="french")}
    res = _resources(_fetcher(games))
    res.delta_parametric = load_parametric_model(DELTA_PARAMETRIC_PATH)
    out = predict_outcome(_prepare(res), res)

    rd = out.rating_delta
    assert rd is not None
    assert rd.model_type == "parametric_elo"
    assert rd.prediction.player_a.profile_id == 100
    assert rd.prediction.player_b.profile_id == 200
    assert rd.prediction.season == 12
    # Current rating/MMR come straight from the games we already fetched.
    assert rd.prediction.player_a.current_rating == 1500
    assert rd.prediction.player_a.current_mmr == 1500
    # Both hypothetical outcomes scored; winning is never worse than losing.
    a = rd.prediction.player_a
    assert a.delta_if_win is not None and a.delta_if_loss is not None
    assert a.delta_if_win >= a.delta_if_loss


def test_live_outcome_omits_rating_delta_without_a_delta_model():
    games = {100: _games_for(100, 200), 200: _games_for(200, 100)}
    res = _resources(_fetcher(games))  # no delta model loaded
    out = predict_outcome(_prepare(res), res)
    assert out.rating_delta is None
