# AOE4 Outcome Prediction Backend

FastAPI wrapper for the historic-data `aoe4_predict` outcome model.

## Local Setup

Install the repo package and dependencies:

```bash
pip install -e .
```

Required local artifacts:

- `aoe4.duckdb`
- `models/aoe4_predict/lgbm_s10s11s12.txt`
- `models/aoe4_predict/lgbm_s10s11s12_meta.json`

Optional artifacts (enable `/predict/rating-delta`):

- `models/ratings_delta/lgbm_delta.txt` — GBT rating-delta model, the default
  (train with `PYTHONPATH=src python -m ratings_delta.train_gbt`)
- `models/p3_parametric.json` — parametric rating-delta model, a cheap
  fallback selectable per request (saved by
  `PYTHONPATH=src python -m ratings_delta.run_nested`)

If neither is present `/predict/rating-delta` returns 503 and everything else
still works; if only one is present it serves that one.

Run the API from the repo root:

```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

## Docker

Build the backend image from the repo root:

```bash
docker build -t aoe4-backend:local .
```

Run it with the local DB and model artifacts mounted read-only:

```bash
docker run --rm \
  -p 8000:8000 \
  -v "$PWD/aoe4.duckdb:/artifacts/aoe4.duckdb:ro" \
  -v "$PWD/models:/artifacts/models:ro" \
  -e AOE4_CORS_ORIGINS=http://localhost:5173 \
  aoe4-backend:local
```

Or use Compose:

```bash
docker compose up --build
```

Verify the running container:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model-info
```

Smoke test a prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict/outcome \
  -H 'Content-Type: application/json' \
  -d '{
    "player_a_id": 1270139,
    "player_b_id": 11513473,
    "civ_a": "english",
    "civ_b": "tughlaq_dynasty",
    "map_name": "Hill and Dale"
  }'
```

## Configuration

Environment variables:

- `AOE4_DB_PATH`, default `./aoe4.duckdb`
- `AOE4_MODEL_PATH`, default `./models/aoe4_predict/lgbm_s10s11s12.txt`
- `AOE4_MODEL_META_PATH`, default `./models/aoe4_predict/lgbm_s10s11s12_meta.json`
- `AOE4_INCLUDE_FEATURES`, default `false`
- `AOE4_CORS_ORIGINS`, comma-separated allowed origins
- `AOE4_CORS_ORIGIN_REGEX`, default allows local dev origins matching `http://localhost:<port>` or `http://127.0.0.1:<port>`; set to an empty string to disable regex-based CORS
- `AOE4_RATE_LIMIT_PER_MINUTE`, default `60`
- `AOE4_MODEL_VERSION`, default `lgbm_s10s11s12`
- `AOE4_DATA_VERSION`, default `aoe4_duckdb_local`
- `AOE4_DELTA_MODEL_PATH`, default `./models/ratings_delta/lgbm_delta.txt`
- `AOE4_DELTA_MODEL_VERSION`, default `lgbm_delta`
- `AOE4_DELTA_PARAMETRIC_PATH`, default `./models/p3_parametric.json`
- `AOE4_DELTA_PARAMETRIC_VERSION`, default `p3_parametric`

## Endpoints

- `GET /health`
- `GET /metadata`
- `GET /model-info`
- `POST /predict/outcome`
- `POST /predict/rating-delta`

**Full request/response documentation: [API.md](API.md)** (also served live at
`/docs` and `/redoc`).

Quick examples:

```bash
curl -X POST http://127.0.0.1:8000/predict/outcome \
  -H 'Content-Type: application/json' \
  -d '{
    "player_a_id": 1270139,
    "player_b_id": 21150142,
    "civ_a": "english",
    "civ_b": "delhi_sultanate",
    "map_name": "Hill and Dale"
  }'

# Conditional rank-point deltas for both players (GBT by default;
# pass "model": "parametric" for the cheap closed-form model)
curl -X POST http://127.0.0.1:8000/predict/rating-delta \
  -H 'Content-Type: application/json' \
  -d '{
    "player_a_id": 3507399,
    "player_b_id": 6914972
  }'
```

## Deployment Notes

The backend is read-only. It does not ingest data, retrain models, mutate
DuckDB, or call external APIs.

For Docker, keep the image to code plus dependencies. Mount the DB and model
artifacts as volumes. Do not bake the 6GB DuckDB into the image except for a
frozen demo.

The built-in rate limiter is process-local and intended for single-process v1
deployments. Use Redis-backed or proxy-level rate limiting for multi-worker or
public deployments.
