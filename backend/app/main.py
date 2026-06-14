from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .middleware import InMemoryRateLimitMiddleware
from .resources import AppResources, Settings, load_resources
from .schemas import (
    HealthResponse,
    MetadataResponse,
    ModelInfoResponse,
    OutcomePredictionRequest,
    OutcomePredictionResponse,
    RatingDeltaRequest,
    RatingDeltaResponse,
)
from .services.delta import predict_rating_delta
from .services.outcome import predict_outcome
from .validation import prepare_outcome_request

logger = logging.getLogger("aoe4.backend")


def create_app(
    settings: Settings | None = None,
    resources: AppResources | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.resources = resources or load_resources(settings)
        yield

    app = FastAPI(
        title="AOE4 Historic Outcome Prediction API",
        version="0.1.0",
        lifespan=lifespan,
    )

    active_settings = settings or (resources.settings if resources else Settings.from_env())
    if active_settings.cors_origins or active_settings.cors_origin_regex:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(active_settings.cors_origins),
            allow_origin_regex=active_settings.cors_origin_regex,
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )
    app.add_middleware(
        InMemoryRateLimitMiddleware,
        requests_per_minute=active_settings.rate_limit_per_minute,
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request method=%s path=%s status=%s latency_ms=%.1f",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        state = _resources(app)
        return HealthResponse(
            status="ok",
            model_loaded=state.model is not None,
            model_meta_loaded=bool(state.meta),
            db_readable=state.settings.db_path.exists(),
            model_version=state.settings.model_version,
            data_version=state.settings.data_version,
            delta_model_loaded=state.delta_model is not None,
            delta_parametric_loaded=state.delta_parametric is not None,
        )

    @app.get("/metadata", response_model=MetadataResponse)
    def metadata() -> MetadataResponse:
        state = _resources(app)
        return MetadataResponse(
            trained_civs=state.trained_categories.get("civ_a", []),
            trained_maps=state.trained_categories.get("map", []),
            trained_patches=state.trained_categories.get("patch", []),
            trained_seasons=state.trained_categories.get("season", []),
            db_civs=state.db_metadata.get("db_civs", []),
            db_maps=state.db_metadata.get("db_maps", []),
            latest_patch=state.db_metadata.get("latest_patch"),
            latest_season=state.db_metadata.get("latest_season"),
        )

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        state = _resources(app)
        return ModelInfoResponse(
            model_version=state.settings.model_version,
            model_type="lightgbm",
            data_version=state.settings.data_version,
            feature_count=len(state.meta.get("feature_cols", [])),
            categorical_features=state.meta.get("cat_features", []),
            training_window=state.meta.get("split", {}),
            metrics=state.meta.get("metrics", {}),
            reference_temporal_metrics=state.meta.get("reference_temporal_metrics"),
            reference_temporal_split=state.meta.get("reference_temporal_split"),
            artifacts={
                "db_path": str(state.settings.db_path),
                "model_path": str(state.settings.model_path),
                "model_meta_path": str(state.settings.model_meta_path),
            },
            delta_models={
                "gbt": {
                    "loaded": state.delta_model is not None,
                    "model_version": state.settings.delta_model_version,
                    "model_type": "lightgbm",
                    "rounding": "regular (0.5 threshold, half away from zero)",
                    "path": str(state.settings.delta_model_path),
                },
                "parametric": {
                    "loaded": state.delta_parametric is not None,
                    "model_version": state.settings.delta_parametric_version,
                    "model_type": "parametric_elo",
                    "rounding": "floor",
                    "path": str(state.settings.delta_parametric_path),
                },
            },
        )

    @app.post("/predict/outcome", response_model=OutcomePredictionResponse)
    def outcome(request: OutcomePredictionRequest) -> OutcomePredictionResponse:
        state = _resources(app)
        prepared = prepare_outcome_request(request, state)
        try:
            return predict_outcome(prepared, state)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("outcome prediction failed")
            raise HTTPException(status_code=500, detail="outcome prediction failed") from exc

    @app.post("/predict/rating-delta", response_model=RatingDeltaResponse)
    def rating_delta(request: RatingDeltaRequest) -> RatingDeltaResponse:
        state = _resources(app)
        try:
            return predict_rating_delta(request, state)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("rating-delta prediction failed")
            raise HTTPException(status_code=500, detail="rating-delta prediction failed") from exc

    return app


def _resources(app: FastAPI) -> AppResources:
    resources = getattr(app.state, "resources", None)
    if resources is None:
        raise HTTPException(status_code=503, detail="resources are not loaded")
    return resources


app = create_app()
