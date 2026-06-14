from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .resources import AppResources
from .schemas import OutcomePredictionRequest


@dataclass
class PreparedOutcomeRequest:
    player_a_id: int
    player_b_id: int
    civ_a: str | None
    civ_b: str | None
    map_name: str | None
    before_timestamp: Any
    warnings: list[str] = field(default_factory=list)
    unseen_categories: list[str] = field(default_factory=list)
    normalized_inputs: dict[str, Any] = field(default_factory=dict)


def prepare_outcome_request(
    request: OutcomePredictionRequest,
    resources: AppResources,
) -> PreparedOutcomeRequest:
    warnings: list[str] = []
    unseen: list[str] = []
    normalized: dict[str, Any] = {}

    civ_a = _validate_civ("civ_a", request.civ_a, resources, warnings, unseen, normalized)
    civ_b = _validate_civ("civ_b", request.civ_b, resources, warnings, unseen, normalized)
    map_name = _validate_map("map_name", request.map_name, resources, warnings, unseen, normalized)

    return PreparedOutcomeRequest(
        player_a_id=request.player_a_id,
        player_b_id=request.player_b_id,
        civ_a=civ_a,
        civ_b=civ_b,
        map_name=map_name,
        before_timestamp=request.before_timestamp,
        warnings=warnings,
        unseen_categories=unseen,
        normalized_inputs=normalized,
    )


def _validate_civ(
    field_name: str,
    value: str | None,
    resources: AppResources,
    warnings: list[str],
    unseen: list[str],
    normalized: dict[str, Any],
) -> str | None:
    if value is None:
        return None

    db_civs = set(resources.db_metadata.get("db_civs", []))
    trained_civs = set(resources.trained_categories.get(field_name, []))
    if value not in db_civs and value not in trained_civs:
        warnings.append(f"{field_name}={value!r} is unknown; using no-civ fallback.")
        unseen.append(field_name)
        normalized[field_name] = None
        return None
    if value not in trained_civs:
        warnings.append(f"{field_name}={value!r} was not seen during model training.")
        unseen.append(field_name)
    return value


def _validate_map(
    field_name: str,
    value: str | None,
    resources: AppResources,
    warnings: list[str],
    unseen: list[str],
    normalized: dict[str, Any],
) -> str | None:
    if value is None:
        return None

    db_maps = set(resources.db_metadata.get("db_maps", []))
    trained_maps = set(resources.trained_categories.get("map", []))
    if value not in db_maps and value not in trained_maps:
        warnings.append(f"{field_name}={value!r} is unknown; using no-map fallback.")
        unseen.append(field_name)
        normalized[field_name] = None
        return None
    if value not in trained_maps:
        warnings.append(f"{field_name}={value!r} was not seen during model training.")
        unseen.append(field_name)
    return value
