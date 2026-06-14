from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from aoe4_predict.db import GAMES_DDL, PARTICIPANTS_DDL
from backend.app.main import create_app
from backend.app.resources import DEFAULT_CORS_ORIGIN_REGEX, AppResources, Settings
from backend.app.schemas import RatingDeltaRequest
from backend.app.services.delta import predict_rating_delta
from ratings_delta.live import (
    get_live_player_state,
    predict_conditional_deltas,
    round_floor,
    round_regular,
)
from ratings_delta.parametric import P3Model


# ── live feature builder + conditional prediction ─────────────────────────────


@pytest.fixture()
def conn():
    conn = duckdb.connect(":memory:")
    conn.execute(GAMES_DDL)
    conn.execute(PARTICIPANTS_DDL)

    # Player 1: 12 games this season (all wins), MMR present.
    # Player 2: 3 games this season (all wins), no MMR (visible rating only).
    # Player 1's opponents in filler games are throwaway IDs (900+).
    for i in range(12):
        game_id = 100 + i
        conn.execute(
            "INSERT INTO games (game_id, started_at, kind, season, patch, map) "
            "VALUES (?, ?, 'rm_1v1', 12, '15.4', 'Dry Arabia')",
            [game_id, datetime(2026, 5, 1 + i, 12, 0)],
        )
        conn.execute(
            """INSERT INTO participants (game_id, profile_id, result, rating, rating_diff, mmr, mmr_diff)
               VALUES (?, 1, true, ?, 10, ?, 12)""",
            [game_id, 1200 + i * 10, 1300 + i * 12],
        )
        conn.execute(
            """INSERT INTO participants (game_id, profile_id, result, rating, rating_diff, mmr, mmr_diff)
               VALUES (?, ?, false, 1100, -10, 1150, -12)""",
            [game_id, 900 + i],
        )
    for i in range(3):
        game_id = 200 + i
        conn.execute(
            "INSERT INTO games (game_id, started_at, kind, season, patch, map) "
            "VALUES (?, ?, 'rm_1v1', 12, '15.4', 'Dry Arabia')",
            [game_id, datetime(2026, 5, 20 + i, 12, 0)],
        )
        conn.execute(
            """INSERT INTO participants (game_id, profile_id, result, rating, rating_diff, mmr, mmr_diff)
               VALUES (?, 2, true, ?, 15, NULL, NULL)""",
            [game_id, 1400 + i * 15],
        )
        conn.execute(
            """INSERT INTO participants (game_id, profile_id, result, rating, rating_diff, mmr, mmr_diff)
               VALUES (?, ?, false, 1380, -15, 1390, -14)""",
            [game_id, 950 + i],
        )
    yield conn
    conn.close()


def test_live_player_state_uses_post_match_values(conn):
    state = get_live_player_state(conn, 1, season=12)

    assert state["rating"] == 1310 + 10  # last pre-match rating + diff
    assert state["mmr"] == 1432 + 12
    assert state["games_lifetime"] == 12
    assert state["games_this_season"] == 12
    assert state["current_streak"] == 12  # twelve straight wins
    assert state["recent_wr_10"] == 1.0
    assert state["recent_wr_20"] == 1.0


def test_live_player_state_respects_before_timestamp(conn):
    state = get_live_player_state(conn, 1, season=12, before_timestamp=datetime(2026, 5, 3))

    assert state["games_this_season"] == 2
    assert state["rating"] == 1210 + 10
    assert state["current_streak"] == 2


def test_live_player_state_without_history(conn):
    state = get_live_player_state(conn, 999_999, season=12)

    assert state["rating"] is None
    assert state["mmr"] is None
    assert state["games_lifetime"] == 0
    assert state["games_this_season"] == 0
    assert state["current_streak"] == 0
    assert state["recent_wr_10"] is None


def test_rounding_rules():
    assert round_regular(23.5) == 24
    assert round_regular(23.49) == 23
    assert round_regular(-23.5) == -24  # half away from zero
    assert round_regular(-23.4) == -23
    assert round_floor(23.9) == 23
    assert round_floor(-23.1) == -24


class _StubBooster:
    """Stands in for a LightGBM booster: returns fixed raw predictions."""

    def predict(self, X):
        return np.array([10.6, -10.6, 3.5, -3.4])


def test_gbt_path_uses_regular_rounding(conn):
    raw = predict_conditional_deltas(_StubBooster(), conn, player_a_id=1, player_b_id=2)

    assert raw["player_a"]["delta_if_win"] == 11
    assert raw["player_a"]["delta_if_loss"] == -11
    assert raw["player_b"]["delta_if_win"] == 4
    assert raw["player_b"]["delta_if_loss"] == -3


def test_p3_path_uses_floor_rounding(conn, monkeypatch):
    monkeypatch.setattr(
        P3Model, "predict", lambda self, df: np.array([10.6, -10.6, 3.5, -3.4])
    )

    raw = predict_conditional_deltas(P3Model(), conn, player_a_id=1, player_b_id=2)

    assert raw["player_a"]["delta_if_win"] == 10
    assert raw["player_a"]["delta_if_loss"] == -11
    assert raw["player_b"]["delta_if_win"] == 3
    assert raw["player_b"]["delta_if_loss"] == -4


def test_conditional_deltas_signs_and_warnings(conn):
    model = P3Model()  # default params: symmetric K=47, no intercept pieces

    raw = predict_conditional_deltas(model, conn, player_a_id=1, player_b_id=2)

    assert raw["season"] == 12
    for side in ("player_a", "player_b"):
        assert isinstance(raw[side]["delta_if_win"], int)  # whole points
        assert isinstance(raw[side]["delta_if_loss"], int)
        assert raw[side]["delta_if_win"] > 0
        assert raw[side]["delta_if_loss"] < 0
    # Higher-rated player B gains no more from a win than underdog A
    # (≤ because floor rounding can collapse a sub-point raw difference).
    assert raw["player_b"]["delta_if_win"] <= raw["player_a"]["delta_if_win"]
    assert any("no MMR history" in w for w in raw["warnings"])  # player B
    assert any("placement-phase" in w for w in raw["warnings"])  # player B, 3 games


def test_conditional_deltas_null_for_unknown_player(conn):
    raw = predict_conditional_deltas(P3Model(), conn, player_a_id=1, player_b_id=999_999)

    assert raw["player_b"]["delta_if_win"] is None
    assert raw["player_b"]["delta_if_loss"] is None
    assert any("cannot be estimated" in w for w in raw["warnings"])


def test_gbt_nulls_for_unknown_player_despite_extrapolation(conn):
    raw = predict_conditional_deltas(_StubBooster(), conn, player_a_id=1, player_b_id=999_999)

    assert raw["player_b"]["delta_if_win"] is None
    assert raw["player_b"]["delta_if_loss"] is None


def test_saved_booster_end_to_end(conn, tmp_path):
    import lightgbm as lgb

    from ratings_delta.model import (
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
        _prepare_X,
        load_lgbm,
        save_lgbm,
    )

    rng = np.random.default_rng(42)
    n = 400
    data = {col: rng.normal(size=n) for col in NUMERIC_FEATURES}
    data["result"] = rng.integers(0, 2, n).astype(float)
    df = pd.DataFrame(data)
    df["season"] = "12"
    df["patch"] = "15.4"
    df["map"] = "Dry Arabia"
    y = np.where(df["result"] == 1, 24.0, -24.0) + rng.normal(scale=2.0, size=n)

    model = lgb.LGBMRegressor(n_estimators=30, min_child_samples=5, verbose=-1)
    model.fit(_prepare_X(df), y, categorical_feature=CATEGORICAL_FEATURES)
    save_lgbm(model, tmp_path / "m.txt", {"best_iteration": 30}, tmp_path / "m.json")
    booster, meta = load_lgbm(tmp_path / "m.txt", tmp_path / "m.json")
    assert meta == {"best_iteration": 30}

    raw = predict_conditional_deltas(booster, conn, player_a_id=1, player_b_id=2)

    for side in ("player_a", "player_b"):
        assert isinstance(raw[side]["delta_if_win"], int)
        assert raw[side]["delta_if_win"] > 0
        assert raw[side]["delta_if_loss"] < 0


# ── backend endpoint ───────────────────────────────────────────────────────────


def _settings(tmp_path: Path) -> Settings:
    db = tmp_path / "aoe4.duckdb"
    model = tmp_path / "model.txt"
    meta = tmp_path / "model_meta.json"
    for path in (db, model, meta):
        path.write_text("ok")
    return Settings(
        db_path=db,
        model_path=model,
        model_meta_path=meta,
        include_features=False,
        cors_origins=(),
        cors_origin_regex=DEFAULT_CORS_ORIGIN_REGEX,
        rate_limit_per_minute=10_000,
        model_version="test_model",
        data_version="test_data",
        delta_model_path=tmp_path / "lgbm_delta.txt",
        delta_model_version="test_delta",
        delta_parametric_path=tmp_path / "p3_parametric.json",
        delta_parametric_version="test_parametric",
    )


def _resources(tmp_path: Path, delta_model=None, delta_parametric=None) -> AppResources:
    return AppResources(
        settings=_settings(tmp_path),
        model=object(),
        meta={"feature_cols": [], "cat_features": []},
        trained_categories={},
        db_metadata={},
        loaded_at=datetime.now(timezone.utc),
        delta_model=delta_model,
        delta_parametric=delta_parametric,
    )


def test_app_exposes_rating_delta_route(tmp_path):
    app = create_app(resources=_resources(tmp_path))
    routes = {route.path for route in app.routes}

    assert "/predict/rating-delta" in routes


def test_request_schema_rejects_same_player():
    with pytest.raises(ValidationError):
        RatingDeltaRequest(player_a_id=5, player_b_id=5)


def test_request_schema_normalizes_empty_map():
    request = RatingDeltaRequest(player_a_id=1, player_b_id=2, map_name="  ")

    assert request.map_name is None


def test_endpoint_returns_503_when_no_delta_model_loaded(tmp_path):
    from fastapi.testclient import TestClient

    app = create_app(resources=_resources(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/predict/rating-delta",
            json={"player_a_id": 1, "player_b_id": 2},
        )

    assert response.status_code == 503


def test_endpoint_returns_503_when_requested_model_missing(tmp_path):
    from fastapi.testclient import TestClient

    # GBT loaded, parametric explicitly requested but absent.
    app = create_app(resources=_resources(tmp_path, delta_model=object()))
    with TestClient(app) as client:
        response = client.post(
            "/predict/rating-delta",
            json={"player_a_id": 1, "player_b_id": 2, "model": "parametric"},
        )

    assert response.status_code == 503
    assert "parametric" in response.json()["detail"]


def test_request_schema_rejects_unknown_model():
    with pytest.raises(ValidationError):
        RatingDeltaRequest(player_a_id=1, player_b_id=2, model="xgboost")


def test_model_selection_prefers_gbt_and_falls_back(tmp_path):
    from backend.app.services.delta import select_delta_model

    gbt, parametric = object(), object()

    both = _resources(tmp_path, delta_model=gbt, delta_parametric=parametric)
    assert select_delta_model(None, both) == (gbt, "test_delta", "lightgbm")
    assert select_delta_model("parametric", both) == (parametric, "test_parametric", "parametric_elo")

    only_parametric = _resources(tmp_path, delta_parametric=parametric)
    assert select_delta_model(None, only_parametric) == (
        parametric,
        "test_parametric",
        "parametric_elo",
    )


def test_predict_rating_delta_response_contract(tmp_path, monkeypatch):
    raw = {
        "season": 12,
        "player_a": {
            "profile_id": 1,
            "rating": 1320,
            "mmr": 1444,
            "games_this_season": 12,
            "delta_if_win": 25,
            "delta_if_loss": -22,
        },
        "player_b": {
            "profile_id": 2,
            "rating": 1445,
            "mmr": None,
            "games_this_season": 3,
            "delta_if_win": 22,
            "delta_if_loss": -25,
        },
        "warnings": ["Player B has no MMR history; delta uses visible-rating fallback."],
    }

    class _FakeConn:
        def close(self):
            pass

    monkeypatch.setattr("backend.app.services.delta.get_conn", lambda *a, **k: _FakeConn())
    monkeypatch.setattr(
        "backend.app.services.delta.predict_conditional_deltas", lambda **kwargs: raw
    )

    resources = _resources(tmp_path, delta_model=object(), delta_parametric=object())
    response = predict_rating_delta(
        RatingDeltaRequest(player_a_id=1, player_b_id=2, map_name="Dry Arabia"), resources
    )

    assert response.request_id
    assert response.prediction.season == 12
    assert response.prediction.player_a.delta_if_win == 25
    assert response.prediction.player_a.delta_if_loss == -22
    assert response.prediction.player_b.current_mmr is None
    assert response.inputs.map_name == "Dry Arabia"
    assert response.inputs.requested_model is None
    assert response.model.model_version == "test_delta"
    assert response.model.model_type == "lightgbm"
    assert response.data_quality.warnings == raw["warnings"]

    parametric_response = predict_rating_delta(
        RatingDeltaRequest(player_a_id=1, player_b_id=2, model="parametric"), resources
    )
    assert parametric_response.inputs.requested_model == "parametric"
    assert parametric_response.model.model_version == "test_parametric"
    assert parametric_response.model.model_type == "parametric_elo"
