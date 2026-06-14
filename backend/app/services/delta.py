from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import HTTPException

from aoe4_predict.db import get_conn
from ratings_delta.live import predict_conditional_deltas

from ..resources import AppResources
from ..schemas import (
    ModelPayload,
    PlayerDeltaPayload,
    RatingDeltaInputsPayload,
    RatingDeltaPrediction,
    RatingDeltaQualityPayload,
    RatingDeltaRequest,
    RatingDeltaResponse,
)


def select_delta_model(
    requested: str | None,
    resources: AppResources,
) -> tuple[Any, str, str]:
    """Resolve the delta model to serve: (model, model_version, model_type).

    `requested` is "gbt", "parametric", or None. When None, the GBT is
    preferred and the parametric model is the fallback, so a deployment can
    drop the GBT artifact and still serve deltas from the cheap model.
    """
    settings = resources.settings
    gbt = (resources.delta_model, settings.delta_model_version, "lightgbm")
    parametric = (resources.delta_parametric, settings.delta_parametric_version, "parametric_elo")

    if requested == "gbt":
        candidates = [gbt]
    elif requested == "parametric":
        candidates = [parametric]
    else:
        candidates = [gbt, parametric]

    for model, version, model_type in candidates:
        if model is not None:
            return model, version, model_type

    detail = (
        f"requested rating-delta model {requested!r} is not loaded"
        if requested
        else "no rating-delta model is loaded"
    )
    raise HTTPException(status_code=503, detail=detail)


def predict_rating_delta(
    request: RatingDeltaRequest,
    resources: AppResources,
) -> RatingDeltaResponse:
    model, model_version, model_type = select_delta_model(request.model, resources)

    conn = get_conn(resources.settings.db_path, read_only=True)
    try:
        raw = predict_conditional_deltas(
            model=model,
            conn=conn,
            player_a_id=request.player_a_id,
            player_b_id=request.player_b_id,
            before_timestamp=request.before_timestamp,
            map_name=request.map_name,
        )
    finally:
        conn.close()

    return RatingDeltaResponse(
        request_id=str(uuid4()),
        prediction_timestamp=datetime.now(timezone.utc),
        prediction=RatingDeltaPrediction(
            player_a=_player_payload(raw["player_a"]),
            player_b=_player_payload(raw["player_b"]),
            season=raw["season"],
        ),
        inputs=RatingDeltaInputsPayload(
            map_name=request.map_name,
            before_timestamp=request.before_timestamp,
            requested_model=request.model,
        ),
        data_quality=RatingDeltaQualityPayload(warnings=raw["warnings"]),
        model=ModelPayload(
            model_version=model_version,
            model_type=model_type,
            data_version=resources.settings.data_version,
        ),
    )


def _player_payload(state: dict[str, Any]) -> PlayerDeltaPayload:
    return PlayerDeltaPayload(
        profile_id=state["profile_id"],
        current_rating=state["rating"],
        current_mmr=state["mmr"],
        games_this_season=state["games_this_season"],
        delta_if_win=state["delta_if_win"],
        delta_if_loss=state["delta_if_loss"],
    )
