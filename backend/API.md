# AOE4 Backend API Reference

FastAPI service serving two prediction models over the local DuckDB match
history:

- **Outcome model** — LightGBM classifier predicting the probability that
  player A beats player B (`/predict/outcome`).
- **Rating-delta models** — predict the ranked-points change for each player
  conditional on each outcome (`/predict/rating-delta`). Two interchangeable
  models are supported (see [Rating-delta models](#rating-delta-models)).

Base URL (local default): `http://127.0.0.1:8000`. All bodies are JSON.
Interactive OpenAPI docs are served at `/docs` (Swagger UI) and `/redoc`.

## General behavior

- **Authentication**: none. The service is read-only and intended for local /
  single-tenant deployments.
- **Rate limiting**: in-memory, per client IP, `AOE4_RATE_LIMIT_PER_MINUTE`
  requests per rolling 60 s (default 60). Exceeding it returns
  `429 {"detail": "rate limit exceeded"}`. The limiter is process-local; use a
  proxy-level limiter for multi-worker deployments.
- **CORS**: `GET`/`POST`/`OPTIONS` allowed for origins in `AOE4_CORS_ORIGINS`
  and/or matching `AOE4_CORS_ORIGIN_REGEX` (default: localhost on any port).
- **Errors** (all endpoints):
  | Status | Meaning |
  |---|---|
  | `422` | Request validation failed (Pydantic detail in body) |
  | `429` | Rate limit exceeded |
  | `500` | Prediction failed unexpectedly (logged server-side) |
  | `503` | Resources not loaded, or the requested delta model is unavailable |

---

## GET /health

Liveness/readiness probe. Always `200` once the app has started.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_meta_loaded": true,
  "db_readable": true,
  "model_version": "lgbm_s10s11s12",
  "data_version": "aoe4_duckdb_local",
  "delta_model_loaded": true,
  "delta_parametric_loaded": true
}
```

| Field | Type | Meaning |
|---|---|---|
| `status` | `"ok"` | App is up |
| `model_loaded` | bool | Outcome model loaded |
| `model_meta_loaded` | bool | Outcome model metadata loaded |
| `db_readable` | bool | DuckDB file exists at the configured path |
| `model_version` / `data_version` | string | Configured version labels |
| `delta_model_loaded` | bool | GBT rating-delta model loaded |
| `delta_parametric_loaded` | bool | Parametric rating-delta model loaded |

---

## GET /metadata

Vocabulary the frontend can use to populate pickers, plus the most recent
patch/season seen in the DB.

| Field | Type | Meaning |
|---|---|---|
| `trained_civs` / `trained_maps` / `trained_patches` / `trained_seasons` | list | Category values the outcome model was trained on |
| `db_civs` / `db_maps` | list[str] | Distinct values present in the DB (may exceed trained sets after a patch) |
| `latest_patch` | str \| null | Patch of the most recent game in the DB |
| `latest_season` | int \| null | Season of the most recent game in the DB |

A value present in `db_*` but missing from `trained_*` is accepted by the
predict endpoints but flagged as an unseen category (see data quality below).

---

## GET /model-info

Outcome-model details plus a summary of both rating-delta models.

| Field | Type | Meaning |
|---|---|---|
| `model_version` / `model_type` / `data_version` | string | Outcome model identity (`model_type` is always `"lightgbm"`) |
| `feature_count` | int | Number of outcome-model features |
| `categorical_features` | list[str] | Outcome-model categorical columns |
| `training_window` | object | Temporal split boundaries from training |
| `metrics` | object | Stored training/validation metrics |
| `reference_temporal_metrics` / `reference_temporal_split` | object \| null | Optional reference-run metrics |
| `artifacts` | object | Resolved paths of DB / model / meta files |
| `delta_models` | object | Per-delta-model summary, keyed `"gbt"` and `"parametric"`; each entry has `loaded`, `model_version`, `model_type`, `rounding`, `path` |

---

## POST /predict/outcome

Probability that player A beats player B, from historic data only.

### Request

```json
{
  "player_a_id": 1270139,
  "player_b_id": 21150142,
  "civ_a": "english",
  "civ_b": "delhi_sultanate",
  "map_name": "Hill and Dale",
  "before_timestamp": "2026-05-01T00:00:00Z"
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `player_a_id` | int > 0 | yes | Relic profile ID; must differ from `player_b_id` |
| `player_b_id` | int > 0 | yes | |
| `civ_a`, `civ_b` | str \| null | no | Lower-cased automatically; empty/whitespace → null. Unknown values fall back to the no-civ prediction with a warning |
| `map_name` | str \| null | no | Case-sensitive; unknown values fall back to no-map with a warning |
| `before_timestamp` | ISO 8601 \| null | no | Score the matchup as of this instant: all player-history queries only use games strictly before it (leak-free historical scoring). Omit for "now" |

### Response

```json
{
  "request_id": "…uuid…",
  "prediction_timestamp": "2026-06-12T21:43:18Z",
  "prediction": {
    "player_a_id": 1270139,
    "player_b_id": 21150142,
    "win_prob_a": 0.62,
    "win_prob_b": 0.38
  },
  "inputs": { "civ_a": "english", "civ_b": "delhi_sultanate", "map_name": "Hill and Dale", "before_timestamp": null },
  "data_quality": {
    "context_level": "full_context",
    "fallback_used": false,
    "warnings": [],
    "imputations": [],
    "unseen_categories": [],
    "normalized_inputs": {}
  },
  "model": { "model_version": "lgbm_s10s11s12", "model_type": "lightgbm", "data_version": "aoe4_duckdb_local" },
  "debug": null
}
```

| Field | Meaning |
|---|---|
| `prediction.win_prob_a` / `win_prob_b` | Win probabilities, sum to 1 |
| `data_quality.context_level` | How much context the prediction used: `"id_only"`, `"map_known"`, `"civ_known"`, or `"full_context"` |
| `data_quality.fallback_used` | True if any input was normalized away, a category was unseen, or cold-start imputation was applied |
| `data_quality.warnings` | Human-readable caveats (sparse history, unseen civ/map/patch/season, cold-start priors, …) |
| `data_quality.imputations` | Cold-start imputations applied, each with `player`, `feature`, `value`, `method`, `prior_games` |
| `data_quality.unseen_categories` | Which request fields carried values absent from model training |
| `data_quality.normalized_inputs` | Request fields the server replaced (e.g. unknown civ → null) |
| `debug.features` | Raw feature vector; only present when `AOE4_INCLUDE_FEATURES=true` |

---

## POST /predict/rating-delta

Conditional ranked-points change for **both** players under **both** outcomes:
what each player gains if they win and loses if they lose. Uses only
pre-match information — current rating/MMR, game counts, streak, recent
form — never the result, duration, or civ picks; the hypothetical winner
enters only as the conditioning variable.

### Rating-delta models

| Name | `model_type` | Artifact | Rounding | Test MAE¹ | Exact hit¹ | Notes |
|---|---|---|---|---|---|---|
| `gbt` (default) | `lightgbm` | `models/ratings_delta/lgbm_delta.txt` | regular (0.5 threshold, half away from zero) | 0.61 | 61% | Primary model; ~3 MB, 20 features incl. map/patch/season |
| `parametric` | `parametric_elo` | `models/p3_parametric.json` | floor | 1.57 | 40% | Cheap fallback: closed-form modified-Elo formula, ~2 KB JSON, no LightGBM needed at scoring time |

¹ On the S10–S12 temporal test split (1.37 M rows), after each model's rounding rule.

Each model uses the rounding rule that empirically minimizes its error: the
game engine floors its Elo deltas, so the formula-faithful parametric model
floors too, while the GBT learns that bias from data and is best with regular
rounding.

**Selection**: the optional `model` request field picks the model. When
omitted, the GBT is used if loaded, otherwise the parametric model — so a
deployment can mount only `p3_parametric.json` and still serve deltas.
Requesting a model that is not loaded returns `503`.

### Request

```json
{
  "player_a_id": 3507399,
  "player_b_id": 6914972,
  "map_name": "Dry Arabia",
  "before_timestamp": null,
  "model": "gbt"
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `player_a_id` | int > 0 | yes | Must differ from `player_b_id` |
| `player_b_id` | int > 0 | yes | |
| `map_name` | str \| null | no | GBT categorical feature; ignored by the parametric model. Empty/whitespace → null |
| `before_timestamp` | ISO 8601 \| null | no | Score as of this instant (same leak-free semantics as the outcome endpoint) |
| `model` | `"gbt"` \| `"parametric"` \| null | no | Model selection; null → prefer GBT, fall back to parametric |

### Response

```json
{
  "request_id": "…uuid…",
  "prediction_timestamp": "2026-06-12T21:43:18Z",
  "prediction": {
    "player_a": {
      "profile_id": 3507399,
      "current_rating": 1408,
      "current_mmr": 1449,
      "games_this_season": 197,
      "delta_if_win": 24,
      "delta_if_loss": -23
    },
    "player_b": {
      "profile_id": 6914972,
      "current_rating": 1386,
      "current_mmr": 1455,
      "games_this_season": 495,
      "delta_if_win": 24,
      "delta_if_loss": -23
    },
    "season": 12
  },
  "inputs": { "map_name": null, "before_timestamp": null, "requested_model": null },
  "data_quality": { "warnings": [] },
  "model": { "model_version": "lgbm_delta", "model_type": "lightgbm", "data_version": "aoe4_duckdb_local" }
}
```

| Field | Meaning |
|---|---|
| `prediction.player_a` / `player_b` | One block per player |
| `…current_rating` | Visible ladder rating going into the next game (post-match value of the player's most recent game); null if the player has no rated games |
| `…current_mmr` | Hidden matchmaking rating, same semantics; null if never observed |
| `…games_this_season` | Rated 1v1 games already played in `prediction.season` |
| `…delta_if_win` | Whole-point rating change if this player wins; null when the player has no rating or MMR history at all |
| `…delta_if_loss` | Whole-point rating change if this player loses (negative); null as above |
| `prediction.season` | Season the prediction applies to (latest in DB, or as of `before_timestamp`) |
| `inputs.requested_model` | Echo of the request's `model` field (null = automatic) |
| `model.model_version` / `model_type` | Which delta model actually served the request |
| `data_quality.warnings` | See below |

Note: `delta_if_win`/`delta_if_loss` are independent per player; the pair
(`player_a.delta_if_win`, `player_b.delta_if_loss`) describes the "A wins"
scenario and (`player_a.delta_if_loss`, `player_b.delta_if_win`) the "B wins"
scenario.

### Warnings emitted

| Warning | Trigger |
|---|---|
| `… has no rating or MMR history; rating delta cannot be estimated.` | Player absent from the DB → that player's deltas are null |
| `… has no MMR history; delta uses visible-rating fallback.` | Hidden MMR never observed; visible rating substitutes (predictions for the *opponent* of such a player are measurably less accurate) |
| `… has played N games this season (< 10); placement-phase deltas are more volatile.` | Player still in the high-K placement phase |

### Examples

```bash
# Default (GBT)
curl -X POST http://127.0.0.1:8000/predict/rating-delta \
  -H 'Content-Type: application/json' \
  -d '{"player_a_id": 3507399, "player_b_id": 6914972}'

# Cheap parametric model, with map context ignored
curl -X POST http://127.0.0.1:8000/predict/rating-delta \
  -H 'Content-Type: application/json' \
  -d '{"player_a_id": 3507399, "player_b_id": 6914972, "model": "parametric"}'

# Historical what-if as of May 1st
curl -X POST http://127.0.0.1:8000/predict/rating-delta \
  -H 'Content-Type: application/json' \
  -d '{"player_a_id": 3507399, "player_b_id": 6914972, "before_timestamp": "2026-05-01T00:00:00Z"}'
```

---

## Configuration

All settings are environment variables read at startup.

| Variable | Default | Purpose |
|---|---|---|
| `AOE4_DB_PATH` | `./aoe4.duckdb` | DuckDB database (opened read-only) |
| `AOE4_MODEL_PATH` | `./models/aoe4_predict/lgbm_s10s11s12.txt` | Outcome model (required) |
| `AOE4_MODEL_META_PATH` | `./models/aoe4_predict/lgbm_s10s11s12_meta.json` | Outcome model metadata (required) |
| `AOE4_DELTA_MODEL_PATH` | `./models/ratings_delta/lgbm_delta.txt` | GBT delta model (optional) |
| `AOE4_DELTA_MODEL_VERSION` | `lgbm_delta` | Version label reported for the GBT |
| `AOE4_DELTA_PARAMETRIC_PATH` | `./models/p3_parametric.json` | Parametric delta model (optional) |
| `AOE4_DELTA_PARAMETRIC_VERSION` | `p3_parametric` | Version label reported for the parametric model |
| `AOE4_INCLUDE_FEATURES` | `false` | Include the raw feature vector in outcome responses (`debug.features`) |
| `AOE4_CORS_ORIGINS` | _(empty)_ | Comma-separated allowed origins |
| `AOE4_CORS_ORIGIN_REGEX` | localhost regex | Regex-based CORS allowlist; empty string disables |
| `AOE4_RATE_LIMIT_PER_MINUTE` | `60` | Per-IP request budget; `0` disables |
| `AOE4_MODEL_VERSION` | `lgbm_s10s11s12` | Outcome model version label |
| `AOE4_DATA_VERSION` | `aoe4_duckdb_local` | Data version label |

Missing **outcome** artifacts fail startup. Missing **delta** artifacts only
disable `/predict/rating-delta` (the endpoint returns `503`); everything else
keeps working. Retrain/regenerate the delta artifacts with:

```bash
PYTHONPATH=src python -m ratings_delta.train_gbt     # → models/ratings_delta/lgbm_delta.txt
PYTHONPATH=src python -m ratings_delta.run_nested    # → models/p3_parametric.json
```
