from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import HTTPException

from aoe4_predict.db import get_conn
from aoe4_predict.inference_api import build_recent_only_features
from aoe4_predict.model import predict_from_features
from aoe4_predict.predict import predict_match
from ratings_delta.live import predict_conditional_deltas_from_games

from ..resources import AppResources
from ..schemas import (
    DataQualityPayload,
    DebugPayload,
    EmbeddedRatingDelta,
    InputsPayload,
    ModelPayload,
    OutcomePredictionResponse,
    PlayerDeltaPayload,
    PredictionPayload,
    RatingDeltaPrediction,
)
from ..validation import PreparedOutcomeRequest
from .aoe4world_client import Aoe4WorldError, Aoe4WorldUnavailable
from .delta import select_delta_model

logger = logging.getLogger("aoe4.backend")


def predict_outcome(
    prepared: PreparedOutcomeRequest,
    resources: AppResources,
) -> OutcomePredictionResponse:
    if resources.settings.feature_source == "api":
        raw = _predict_via_api(prepared, resources)
    else:
        conn = get_conn(resources.settings.db_path, read_only=True)
        try:
            raw = predict_match(
                player_a_id=prepared.player_a_id,
                player_b_id=prepared.player_b_id,
                civ_a=prepared.civ_a,
                civ_b=prepared.civ_b,
                map_name=prepared.map_name,
                conn=conn,
                model=resources.model,
                meta=resources.meta,
                before_timestamp=prepared.before_timestamp,
            )
        finally:
            conn.close()

    warnings = [*prepared.warnings, *raw.get("warnings", [])]
    patch = raw.get("model_meta", {}).get("patch")
    season = raw.get("model_meta", {}).get("season")
    unseen_categories = list(prepared.unseen_categories)
    _append_model_unseen("patch", patch, resources, warnings, unseen_categories)
    _append_model_unseen("season", season, resources, warnings, unseen_categories)

    fallback_used = bool(
        prepared.normalized_inputs
        or raw.get("imputations")
        or prepared.unseen_categories
        or any("fallback" in warning.lower() for warning in warnings)
    )
    debug = None
    if resources.settings.include_features:
        debug = DebugPayload(features=raw.get("features", {}))

    return OutcomePredictionResponse(
        request_id=str(uuid4()),
        prediction_timestamp=datetime.now(timezone.utc),
        prediction=PredictionPayload(
            player_a_id=raw["player_a_id"],
            player_b_id=raw["player_b_id"],
            win_prob_a=raw["win_prob_a"],
            win_prob_b=raw["win_prob_b"],
        ),
        inputs=InputsPayload(
            civ_a=prepared.civ_a,
            civ_b=prepared.civ_b,
            map_name=prepared.map_name,
            before_timestamp=prepared.before_timestamp,
        ),
        data_quality=DataQualityPayload(
            context_level=raw["context_level"],
            fallback_used=fallback_used,
            warnings=warnings,
            imputations=raw.get("imputations", []),
            unseen_categories=sorted(set(unseen_categories)),
            normalized_inputs=prepared.normalized_inputs,
            data_freshness=raw.get("data_freshness"),
        ),
        model=ModelPayload(
            model_version=resources.settings.model_version,
            model_type="lightgbm",
            data_version=resources.settings.data_version,
        ),
        rating_delta=raw.get("rating_delta"),
        debug=debug,
    )


def _predict_via_api(
    prepared: PreparedOutcomeRequest,
    resources: AppResources,
) -> dict[str, Any]:
    """DB-free outcome via live aoe4world recent-games + cached matchup snapshot."""
    if prepared.before_timestamp is not None:
        raise HTTPException(
            status_code=400,
            detail="before_timestamp (historical scoring) is not supported with the live API feature source.",
        )
    client = resources.aoe4world_client
    pa, pb = prepared.player_a_id, prepared.player_b_id
    try:
        games_a, games_b = client.fetch_both(pa, pb)
        snapshot = client.fetch_matchup_priors()
    except Aoe4WorldUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"aoe4world unavailable: {exc}") from exc
    except Aoe4WorldError as exc:
        raise HTTPException(status_code=502, detail=f"aoe4world response error: {exc}") from exc

    now = datetime.utcnow()
    feat = build_recent_only_features(
        games_a=games_a,
        games_b=games_b,
        profile_a=pa,
        profile_b=pb,
        civ_a=prepared.civ_a,
        civ_b=prepared.civ_b,
        map_name=prepared.map_name,
        matchup_lookup=client.matchup_lookup(snapshot),
        now=now,
    )
    win_prob_a = predict_from_features(resources.model, feat, resources.meta["feature_cols"])

    if prepared.civ_a and prepared.civ_b and prepared.map_name:
        context = "full_context"
    elif prepared.civ_a and prepared.civ_b:
        context = "civ_known"
    elif prepared.map_name:
        context = "map_known"
    else:
        context = "id_only"

    warnings: list[str] = []
    for side, games in (("A", games_a), ("B", games_b)):
        if not games:
            warnings.append(f"Player {side} has no recent aoe4world games — using cold-start priors.")
        elif len(games) < 10:
            warnings.append(f"Player {side} has only {len(games)} recent games — prediction less reliable.")
    if feat.get("missing_skill_a"):
        warnings.append("No MMR or rating found for Player A from aoe4world.")
    if feat.get("missing_skill_b"):
        warnings.append("No MMR or rating found for Player B from aoe4world.")
    for imputation in feat.get("imputations", []):
        warnings.append(
            f"{imputation['player']} skill imputed with {imputation['method'].replace('_', ' ')} "
            f"cold-start prior ({int(imputation['value'])})."
        )
    if context == "id_only":
        warnings.append("No civ or map context provided — prediction is less specific.")

    # Rating-point deltas ride off this same aoe4world fetch: reuse the games we
    # already have instead of a second, DB-bound /predict/rating-delta call.
    rating_delta = _live_rating_delta(
        resources, games_a, games_b, pa, pb, feat.get("season"), feat.get("patch"),
        prepared.map_name, now,
    )

    return {
        "player_a_id": pa,
        "player_b_id": pb,
        "civ_a": prepared.civ_a,
        "civ_b": prepared.civ_b,
        "map_name": prepared.map_name,
        "context_level": context,
        "win_prob_a": round(win_prob_a, 4),
        "win_prob_b": round(1 - win_prob_a, 4),
        "features": feat,
        "imputations": feat.get("imputations", []),
        "prediction_confidence": feat.get("prediction_confidence", "standard"),
        "warnings": warnings,
        "model_meta": {
            "patch": feat.get("patch"),
            "season": feat.get("season"),
            "n_trees": resources.meta.get("n_trees"),
            "valid_auc": resources.meta.get("metrics", {}).get("valid", {}).get("auc"),
        },
        # Live data freshness for auditability.
        "data_freshness": {
            "player_a_games": len(games_a),
            "player_b_games": len(games_b),
            "player_a_latest": games_a[0]["started_at"].isoformat() if games_a else None,
            "player_b_latest": games_b[0]["started_at"].isoformat() if games_b else None,
        },
        "rating_delta": rating_delta,
    }


def _live_rating_delta(
    resources: AppResources,
    games_a: list[dict],
    games_b: list[dict],
    player_a_id: int,
    player_b_id: int,
    season: Any,
    patch: Any,
    map_name: str | None,
    now: datetime,
) -> EmbeddedRatingDelta | None:
    """Best-effort rating-point deltas from the already-fetched recent games.

    Returns None (rather than failing the outcome) when no delta model is loaded
    or scoring raises, so the win-probability prediction is never blocked by the
    supplementary rating estimate.
    """
    try:
        model, model_version, model_type = select_delta_model(None, resources)
    except HTTPException:
        return None
    try:
        raw = predict_conditional_deltas_from_games(
            model=model,
            games_a=games_a,
            games_b=games_b,
            player_a_id=player_a_id,
            player_b_id=player_b_id,
            season=season,
            patch=patch,
            map_name=map_name,
            now=now,
        )
    except Exception:
        logger.exception("live rating-delta computation failed")
        return None

    return EmbeddedRatingDelta(
        prediction=RatingDeltaPrediction(
            player_a=_live_player_payload(raw["player_a"]),
            player_b=_live_player_payload(raw["player_b"]),
            season=raw["season"],
        ),
        warnings=raw["warnings"],
        model_version=model_version,
        model_type=model_type,
    )


def _live_player_payload(state: dict[str, Any]) -> PlayerDeltaPayload:
    return PlayerDeltaPayload(
        profile_id=state["profile_id"],
        current_rating=state["rating"],
        current_mmr=state["mmr"],
        games_this_season=state["games_this_season"],
        delta_if_win=state["delta_if_win"],
        delta_if_loss=state["delta_if_loss"],
    )


def _append_model_unseen(
    field_name: str,
    value: Any,
    resources: AppResources,
    warnings: list[str],
    unseen_categories: list[str],
) -> None:
    if value is None:
        return
    trained = set(resources.trained_categories.get(field_name, []))
    if value not in trained:
        warnings.append(f"{field_name}={value!r} was not seen during model training.")
        unseen_categories.append(field_name)
