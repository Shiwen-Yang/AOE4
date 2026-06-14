from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aoe4_predict.config import BASE_DIR, DB_PATH, MODEL_DIR, MODEL_META_PATH, MODEL_PATH
from aoe4_predict.db import get_conn
from aoe4_predict.model import load_model
from ratings_delta.model import load_lgbm
from ratings_delta.parametric import P3Model

from .services.aoe4world_client import DEFAULT_BASE_URL, DEFAULT_USER_AGENT, Aoe4WorldClient


DEFAULT_CORS_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
DELTA_MODEL_PATH = BASE_DIR / "models" / "ratings_delta" / "lgbm_delta.txt"
DELTA_PARAMETRIC_PATH = BASE_DIR / "models" / "p3_parametric.json"

# DB-free API30 model, served when AOE4_FEATURE_SOURCE=api (the default).
API30_MODEL_PATH = MODEL_DIR / "lgbm_api30_recent_only.txt"
API30_MODEL_META_PATH = MODEL_DIR / "lgbm_api30_recent_only_meta.json"

logger = logging.getLogger("aoe4.backend")


@dataclass(frozen=True)
class Settings:
    db_path: Path
    model_path: Path
    model_meta_path: Path
    include_features: bool
    cors_origins: tuple[str, ...]
    cors_origin_regex: str | None
    rate_limit_per_minute: int
    model_version: str
    data_version: str
    delta_model_path: Path = DELTA_MODEL_PATH
    delta_model_version: str = "lgbm_delta"
    delta_parametric_path: Path = DELTA_PARAMETRIC_PATH
    delta_parametric_version: str = "p3_parametric"
    # Feature source: "api" (live aoe4world, DB-free) or "db" (local DuckDB).
    feature_source: str = "api"
    aoe4world_base_url: str = DEFAULT_BASE_URL
    aoe4world_timeout: float = 8.0
    aoe4world_retries: int = 2
    aoe4world_games_ttl: float = 45.0
    aoe4world_matchups_ttl: float = 6 * 3600.0
    aoe4world_user_agent: str = DEFAULT_USER_AGENT

    @classmethod
    def from_env(cls) -> "Settings":
        feature_source = os.getenv("AOE4_FEATURE_SOURCE", "api").strip().lower()
        api_mode = feature_source == "api"
        default_model = API30_MODEL_PATH if api_mode else MODEL_PATH
        default_meta = API30_MODEL_META_PATH if api_mode else MODEL_META_PATH
        default_version = "lgbm_api30_recent_only" if api_mode else "lgbm_s10s11s12"
        return cls(
            db_path=Path(os.getenv("AOE4_DB_PATH", str(DB_PATH))),
            model_path=Path(os.getenv("AOE4_MODEL_PATH", str(default_model))),
            model_meta_path=Path(os.getenv("AOE4_MODEL_META_PATH", str(default_meta))),
            include_features=_env_bool("AOE4_INCLUDE_FEATURES", default=False),
            cors_origins=_env_csv("AOE4_CORS_ORIGINS"),
            cors_origin_regex=os.getenv("AOE4_CORS_ORIGIN_REGEX", DEFAULT_CORS_ORIGIN_REGEX) or None,
            rate_limit_per_minute=int(os.getenv("AOE4_RATE_LIMIT_PER_MINUTE", "60")),
            model_version=os.getenv("AOE4_MODEL_VERSION", default_version),
            data_version=os.getenv("AOE4_DATA_VERSION", "aoe4world_live" if api_mode else "aoe4_duckdb_local"),
            feature_source=feature_source,
            aoe4world_base_url=os.getenv("AOE4WORLD_BASE_URL", DEFAULT_BASE_URL),
            aoe4world_timeout=float(os.getenv("AOE4WORLD_TIMEOUT", "8")),
            aoe4world_retries=int(os.getenv("AOE4WORLD_RETRIES", "2")),
            aoe4world_games_ttl=float(os.getenv("AOE4WORLD_GAMES_TTL", "45")),
            aoe4world_matchups_ttl=float(os.getenv("AOE4WORLD_MATCHUPS_TTL", str(6 * 3600))),
            aoe4world_user_agent=os.getenv("AOE4WORLD_USER_AGENT", DEFAULT_USER_AGENT),
            delta_model_path=Path(os.getenv("AOE4_DELTA_MODEL_PATH", str(DELTA_MODEL_PATH))),
            delta_model_version=os.getenv("AOE4_DELTA_MODEL_VERSION", "lgbm_delta"),
            delta_parametric_path=Path(
                os.getenv("AOE4_DELTA_PARAMETRIC_PATH", str(DELTA_PARAMETRIC_PATH))
            ),
            delta_parametric_version=os.getenv("AOE4_DELTA_PARAMETRIC_VERSION", "p3_parametric"),
        )


@dataclass
class AppResources:
    settings: Settings
    model: Any
    meta: dict[str, Any]
    trained_categories: dict[str, list[Any]]
    db_metadata: dict[str, Any]
    loaded_at: datetime
    delta_model: Any = None
    delta_parametric: Any = None
    aoe4world_client: Any = None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def validate_artifacts(settings: Settings) -> None:
    # In api mode the DuckDB file is not required (features come from aoe4world).
    required = [settings.model_path, settings.model_meta_path]
    if settings.feature_source != "api":
        required.insert(0, settings.db_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required backend artifact(s): " + ", ".join(missing))


def load_resources(settings: Settings | None = None) -> AppResources:
    settings = settings or Settings.from_env()
    validate_artifacts(settings)
    model, meta = load_model(settings.model_path, settings.model_meta_path)
    trained_categories = load_trained_categories(settings.model_path, meta)

    aoe4world_client = None
    if settings.feature_source == "api":
        # DB-free: civ/map vocab comes from the model's own trained categories
        # (validation.py already prefers those); no DuckDB read at startup.
        db_metadata = {"db_civs": [], "db_maps": [], "latest_patch": None, "latest_season": None}
        aoe4world_client = Aoe4WorldClient(
            base_url=settings.aoe4world_base_url,
            timeout=settings.aoe4world_timeout,
            retries=settings.aoe4world_retries,
            user_agent=settings.aoe4world_user_agent,
            games_ttl=settings.aoe4world_games_ttl,
            matchups_ttl=settings.aoe4world_matchups_ttl,
        )
    else:
        db_metadata = load_db_metadata(settings.db_path)

    return AppResources(
        settings=settings,
        model=model,
        meta=meta,
        trained_categories=trained_categories,
        db_metadata=db_metadata,
        loaded_at=datetime.now(timezone.utc),
        delta_model=load_delta_model(settings.delta_model_path),
        delta_parametric=load_parametric_model(settings.delta_parametric_path),
        aoe4world_client=aoe4world_client,
    )


def load_delta_model(path: Path) -> Any | None:
    """Load the GBT rating-delta booster (primary delta model); None if absent.

    Both delta models are optional so outcome-only deployments keep working;
    /predict/rating-delta returns 503 until at least one artifact is provided.
    """
    if not path.exists():
        logger.warning("GBT rating-delta model not found at %s", path)
        return None
    booster, _ = load_lgbm(path)
    return booster


def load_parametric_model(path: Path) -> P3Model | None:
    """Load the P3 parametric rating-delta model (cheap fallback); None if absent."""
    if not path.exists():
        logger.warning("parametric rating-delta model not found at %s", path)
        return None
    return P3Model.load(path)


def load_trained_categories(model_path: Path, meta: dict[str, Any]) -> dict[str, list[Any]]:
    cat_features = list(meta.get("cat_features", []))
    categories = {name: [] for name in cat_features}
    if not model_path.exists() or not cat_features:
        return categories

    for line in model_path.read_text(errors="ignore").splitlines():
        if not line.startswith("pandas_categorical:"):
            continue
        raw = line.split(":", 1)[1]
        values = json.loads(raw)
        return {
            name: list(values[idx]) if idx < len(values) else []
            for idx, name in enumerate(cat_features)
        }
    return categories


def load_db_metadata(db_path: Path) -> dict[str, Any]:
    conn = get_conn(db_path, read_only=True)
    try:
        db_civs = [
            row[0]
            for row in conn.execute(
                """
                SELECT DISTINCT civilization
                FROM participants
                WHERE civilization IS NOT NULL
                ORDER BY civilization
                """
            ).fetchall()
        ]
        db_maps = [
            row[0]
            for row in conn.execute(
                """
                SELECT DISTINCT map
                FROM games
                WHERE map IS NOT NULL
                ORDER BY map
                """
            ).fetchall()
        ]
        latest = conn.execute(
            """
            SELECT patch, season
            FROM games
            WHERE started_at IS NOT NULL
            ORDER BY started_at DESC
            LIMIT 1
            """
        ).fetchone()
    finally:
        conn.close()

    return {
        "db_civs": db_civs,
        "db_maps": db_maps,
        "latest_patch": latest[0] if latest else None,
        "latest_season": latest[1] if latest else None,
    }
