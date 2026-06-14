from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.main import create_app
from backend.app.resources import DEFAULT_CORS_ORIGIN_REGEX, AppResources, Settings, load_resources
from backend.app.schemas import OutcomePredictionRequest
from backend.app.services.outcome import predict_outcome
from backend.app.validation import prepare_outcome_request


class _FakeConn:
    def close(self):
        pass


def _settings(tmp_path: Path, include_features: bool = False) -> Settings:
    db = tmp_path / "aoe4.duckdb"
    model = tmp_path / "model.txt"
    meta = tmp_path / "model_meta.json"
    for path in (db, model, meta):
        path.write_text("ok")
    return Settings(
        db_path=db,
        model_path=model,
        model_meta_path=meta,
        include_features=include_features,
        cors_origins=(),
        cors_origin_regex=DEFAULT_CORS_ORIGIN_REGEX,
        rate_limit_per_minute=10_000,
        model_version="test_model",
        data_version="test_data",
        feature_source="db",
    )


def _resources(tmp_path: Path, include_features: bool = False) -> AppResources:
    return AppResources(
        settings=_settings(tmp_path, include_features=include_features),
        model=object(),
        meta={
            "feature_cols": ["skill_a", "skill_b"],
            "cat_features": ["civ_a", "civ_b", "map", "patch", "season"],
            "split": {"train_end": "2025-01-01", "valid_end": "2025-02-01"},
            "metrics": {"valid": {"auc": 0.7}},
        },
        trained_categories={
            "civ_a": ["english", "delhi_sultanate"],
            "civ_b": ["english", "delhi_sultanate"],
            "map": ["Hill and Dale"],
            "patch": ["13.0.4343.0"],
            "season": [10],
        },
        db_metadata={
            "db_civs": ["english", "delhi_sultanate", "tughlaq_dynasty"],
            "db_maps": ["Hill and Dale", "Craters"],
            "latest_patch": "15.4.8719.0",
            "latest_season": 12,
        },
        loaded_at=datetime.now(timezone.utc),
    )


def _mock_predict(monkeypatch, *, features=None, warnings=None, imputations=None):
    def fake_get_conn(*args, **kwargs):
        return _FakeConn()

    def fake_predict_match(**kwargs):
        return {
            "player_a_id": kwargs["player_a_id"],
            "player_b_id": kwargs["player_b_id"],
            "civ_a": kwargs.get("civ_a"),
            "civ_b": kwargs.get("civ_b"),
            "map_name": kwargs.get("map_name"),
            "context_level": "full_context"
            if kwargs.get("civ_a") and kwargs.get("civ_b") and kwargs.get("map_name")
            else "id_only",
            "win_prob_a": 0.75,
            "win_prob_b": 0.25,
            "warnings": warnings or [],
            "imputations": imputations or [],
            "features": features or {"skill_a": 1200},
            "model_meta": {"patch": "15.4.8719.0", "season": 12},
        }

    monkeypatch.setattr("backend.app.services.outcome.get_conn", fake_get_conn)
    monkeypatch.setattr("backend.app.services.outcome.predict_match", fake_predict_match)


def test_app_exposes_expected_routes(tmp_path):
    app = create_app(resources=_resources(tmp_path))
    routes = {route.path for route in app.routes}

    assert "/health" in routes
    assert "/metadata" in routes
    assert "/model-info" in routes
    assert "/predict/outcome" in routes


def test_app_configures_localhost_cors_by_default(tmp_path):
    app = create_app(resources=_resources(tmp_path))
    cors = [middleware for middleware in app.user_middleware if middleware.cls.__name__ == "CORSMiddleware"]

    assert cors
    assert cors[0].kwargs["allow_origin_regex"] == DEFAULT_CORS_ORIGIN_REGEX
    assert "POST" in cors[0].kwargs["allow_methods"]


def test_request_schema_rejects_invalid_player_ids_and_same_player():
    with pytest.raises(ValidationError):
        OutcomePredictionRequest(player_a_id=0, player_b_id=2)
    with pytest.raises(ValidationError):
        OutcomePredictionRequest(player_a_id=2, player_b_id=2)


def test_request_schema_normalizes_empty_strings():
    request = OutcomePredictionRequest(
        player_a_id=1,
        player_b_id=2,
        civ_a="",
        civ_b=" English ",
        map_name=" ",
    )

    assert request.civ_a is None
    assert request.civ_b == "english"
    assert request.map_name is None


def test_prepare_unknown_civ_and_map_fall_back_to_none(tmp_path):
    request = OutcomePredictionRequest(
        player_a_id=1,
        player_b_id=2,
        civ_a="not_a_civ",
        map_name="Not A Map",
    )

    prepared = prepare_outcome_request(request, _resources(tmp_path))

    assert prepared.civ_a is None
    assert prepared.map_name is None
    assert "civ_a" in prepared.unseen_categories
    assert "map_name" in prepared.unseen_categories
    assert prepared.normalized_inputs == {"civ_a": None, "map_name": None}


def test_prepare_db_known_but_model_unseen_category_warns(tmp_path):
    request = OutcomePredictionRequest(
        player_a_id=1,
        player_b_id=2,
        civ_b="tughlaq_dynasty",
        map_name="Craters",
    )

    prepared = prepare_outcome_request(request, _resources(tmp_path))

    assert prepared.civ_b == "tughlaq_dynasty"
    assert prepared.map_name == "Craters"
    assert "civ_b" in prepared.unseen_categories
    assert "map_name" in prepared.unseen_categories
    assert any("not seen during model training" in w for w in prepared.warnings)


def test_predict_outcome_response_contract(tmp_path, monkeypatch):
    _mock_predict(monkeypatch)
    resources = _resources(tmp_path)
    request = OutcomePredictionRequest(
        player_a_id=1270139,
        player_b_id=21150142,
        civ_a="english",
        civ_b="delhi_sultanate",
        map_name="Hill and Dale",
    )
    prepared = prepare_outcome_request(request, resources)

    response = predict_outcome(prepared, resources)

    assert response.request_id
    assert response.prediction.win_prob_a == 0.75
    assert response.prediction.win_prob_b == 0.25
    assert response.prediction.win_prob_a + response.prediction.win_prob_b == pytest.approx(1.0)
    assert response.model.model_version == "test_model"
    assert response.debug is None


def test_include_features_exposes_debug_payload(tmp_path, monkeypatch):
    _mock_predict(monkeypatch, features={"skill_a": 1400, "skill_b": 1200})
    resources = _resources(tmp_path, include_features=True)
    prepared = prepare_outcome_request(
        OutcomePredictionRequest(player_a_id=1, player_b_id=2),
        resources,
    )

    response = predict_outcome(prepared, resources)

    assert response.debug is not None
    assert response.debug.features == {"skill_a": 1400, "skill_b": 1200}


def test_cold_start_imputations_are_returned(tmp_path, monkeypatch):
    imputation = {
        "player": "Player B",
        "feature": "skill_b",
        "value": 1000,
        "method": "opponent_aware",
        "prior_games": 5,
    }
    _mock_predict(monkeypatch, warnings=["Prediction uses cold-start priors."], imputations=[imputation])
    resources = _resources(tmp_path)
    prepared = prepare_outcome_request(
        OutcomePredictionRequest(player_a_id=1, player_b_id=2),
        resources,
    )

    response = predict_outcome(prepared, resources)

    assert response.data_quality.fallback_used is True
    assert response.data_quality.imputations == [imputation]
    assert "Prediction uses cold-start priors." in response.data_quality.warnings


def test_startup_fails_when_artifacts_missing(tmp_path):
    settings = Settings(
        db_path=tmp_path / "missing.duckdb",
        model_path=tmp_path / "missing_model.txt",
        model_meta_path=tmp_path / "missing_meta.json",
        include_features=False,
        cors_origins=(),
        cors_origin_regex=DEFAULT_CORS_ORIGIN_REGEX,
        rate_limit_per_minute=60,
        model_version="missing",
        data_version="missing",
    )

    with pytest.raises(FileNotFoundError):
        load_resources(settings)
