from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator


ContextLevel = Literal["full_context", "civ_known", "map_known", "id_only"]
DeltaModelName = Literal["gbt", "parametric"]


class OutcomePredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_a_id: PositiveInt
    player_b_id: PositiveInt
    civ_a: str | None = None
    civ_b: str | None = None
    map_name: str | None = None
    before_timestamp: datetime | None = None

    @field_validator("civ_a", "civ_b", "map_name", mode="before")
    @classmethod
    def clean_optional_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("value must be a string or null")
        cleaned = value.strip()
        return cleaned or None

    @field_validator("civ_a", "civ_b")
    @classmethod
    def normalize_civ_case(cls, value: str | None) -> str | None:
        return value.lower() if value is not None else None

    @model_validator(mode="after")
    def players_must_differ(self) -> "OutcomePredictionRequest":
        if self.player_a_id == self.player_b_id:
            raise ValueError("player_a_id and player_b_id must differ")
        return self


class PredictionPayload(BaseModel):
    player_a_id: int
    player_b_id: int
    win_prob_a: float = Field(ge=0, le=1)
    win_prob_b: float = Field(ge=0, le=1)


class InputsPayload(BaseModel):
    civ_a: str | None
    civ_b: str | None
    map_name: str | None
    before_timestamp: datetime | None


class DataQualityPayload(BaseModel):
    context_level: ContextLevel
    fallback_used: bool
    warnings: list[str]
    imputations: list[dict[str, Any]]
    unseen_categories: list[str] = Field(default_factory=list)
    normalized_inputs: dict[str, Any] = Field(default_factory=dict)
    # Live (api) feature source only: per-player recent-game counts + latest timestamps.
    data_freshness: dict[str, Any] | None = None


class ModelPayload(BaseModel):
    model_version: str
    model_type: str
    data_version: str


class DebugPayload(BaseModel):
    features: dict[str, Any] | None = None


class EmbeddedRatingDelta(BaseModel):
    """Rating-point deltas computed from the same live aoe4world data as the
    outcome prediction (api feature source only); null when no delta model is
    loaded or the live data is unavailable."""

    prediction: "RatingDeltaPrediction"
    warnings: list[str]
    model_version: str
    model_type: str


class OutcomePredictionResponse(BaseModel):
    request_id: str
    prediction_timestamp: datetime
    prediction: PredictionPayload
    inputs: InputsPayload
    data_quality: DataQualityPayload
    model: ModelPayload
    rating_delta: EmbeddedRatingDelta | None = None
    debug: DebugPayload | None = None


class RatingDeltaRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_a_id: PositiveInt
    player_b_id: PositiveInt
    map_name: str | None = None
    before_timestamp: datetime | None = None
    model: DeltaModelName | None = None

    @field_validator("map_name", mode="before")
    @classmethod
    def clean_optional_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("value must be a string or null")
        cleaned = value.strip()
        return cleaned or None

    @model_validator(mode="after")
    def players_must_differ(self) -> "RatingDeltaRequest":
        if self.player_a_id == self.player_b_id:
            raise ValueError("player_a_id and player_b_id must differ")
        return self


class PlayerDeltaPayload(BaseModel):
    profile_id: int
    current_rating: int | None
    current_mmr: int | None
    games_this_season: int
    delta_if_win: int | None
    delta_if_loss: int | None


class RatingDeltaPrediction(BaseModel):
    player_a: PlayerDeltaPayload
    player_b: PlayerDeltaPayload
    season: int | None


class RatingDeltaInputsPayload(BaseModel):
    map_name: str | None
    before_timestamp: datetime | None
    requested_model: DeltaModelName | None


class RatingDeltaQualityPayload(BaseModel):
    warnings: list[str]


class RatingDeltaResponse(BaseModel):
    request_id: str
    prediction_timestamp: datetime
    prediction: RatingDeltaPrediction
    inputs: RatingDeltaInputsPayload
    data_quality: RatingDeltaQualityPayload
    model: ModelPayload


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_meta_loaded: bool
    db_readable: bool
    model_version: str
    data_version: str
    delta_model_loaded: bool = False
    delta_parametric_loaded: bool = False


class MetadataResponse(BaseModel):
    trained_civs: list[Any]
    trained_maps: list[Any]
    trained_patches: list[Any]
    trained_seasons: list[Any]
    db_civs: list[str]
    db_maps: list[str]
    latest_patch: str | None
    latest_season: int | None


class ModelInfoResponse(BaseModel):
    model_version: str
    model_type: str
    data_version: str
    feature_count: int
    categorical_features: list[str]
    training_window: dict[str, Any]
    metrics: dict[str, Any]
    reference_temporal_metrics: dict[str, Any] | None = None
    reference_temporal_split: dict[str, Any] | None = None
    artifacts: dict[str, str]
    delta_models: dict[str, Any] = Field(default_factory=dict)


# EmbeddedRatingDelta forward-references RatingDeltaPrediction (defined above but
# after OutcomePredictionResponse); resolve the reference now that both exist.
EmbeddedRatingDelta.model_rebuild()
OutcomePredictionResponse.model_rebuild()
