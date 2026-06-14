# AOE4

Tools and experiments for Age of Empires IV ranked 1v1 prediction and
analysis.

The repository is organized around a local DuckDB match warehouse
(`aoe4.duckdb`) plus Python packages for pregame outcome prediction,
civilization-choice modeling, rating-update analysis, replay snapshot modeling,
and a small FastAPI prediction service.

## Project Areas

- `aoe4_predict`: pregame RM 1v1 match outcome prediction from historic player,
  civilization, map, patch, and form features.
- `civ_choice`: civilization-choice prediction, including known-player and
  anonymous-opponent variants.
- `ratings_delta`: investigation of visible rating updates and Elo-like rating
  delta formulas.
- `realtime_outcome_prediction`: replay snapshot features for in-game outcome
  prediction.
- `player_clustering`: player civilization preference clustering utilities.
- `backend`: FastAPI wrapper around the historic outcome model.

## Requirements

- Python 3.10+
- Local match database: `aoe4.duckdb`
- For API predictions, trained model artifacts under `models/aoe4_predict/`

Install the repo in editable mode from the repository root:

```bash
python -m pip install -e .
```

The main Python dependencies are declared in `pyproject.toml` and include
DuckDB, pandas, NumPy, LightGBM, FastAPI, and Uvicorn.

## Data and Artifacts

This repo expects large generated artifacts to live locally:

- `aoe4.duckdb`: local DuckDB warehouse for match, participant, metadata, and
  feature tables.
- `data/`: raw, parsed, cached, and feature data.
- `models/`: trained model artifacts.
- `reports/generated/`: generated markdown, CSV, and JSON report outputs.
- `reports/figures/`: generated report figures.
- `lightning_logs/`: local training logs.

These artifacts are intentionally not source-controlled. Static reference data
that should be tracked lives in `metadata/`.

## Common Commands

After editable install, use either `python -m ...` or the console scripts
defined in `pyproject.toml`.

```bash
python -m aoe4_predict --help
python -m civ_choice.run --help
python -m civ_choice.run_anonymous_opponent --help
python -m ratings_delta.run --help
python -m realtime_outcome_prediction --help
```

Equivalent installed commands:

```bash
aoe4-predict --help
civ-choice --help
civ-choice-anonymous-opponent --help
civ-choice-live --help
ratings-delta --help
realtime-outcome-prediction --help
```

## Pregame Outcome Prediction

`aoe4_predict` owns the main historic match outcome workflow.

```bash
# Load JSON.gz match data into DuckDB.
python -m aoe4_predict ingest --data-dir data/raw --seasons 10,11,12

# Load tracked map and patch metadata.
python -m aoe4_predict ingest-metadata --metadata-dir metadata

# Generate a data quality report.
python -m aoe4_predict quality

# Train a LightGBM model.
python -m aoe4_predict train --seasons 10,11,12

# Train with a held-out season and slot-swap augmentation.
python -m aoe4_predict train --seasons 10,11 --test-seasons 12 --symmetric-slots

# Evaluate the saved model against baselines.
python -m aoe4_predict evaluate

# Generate the main analysis report.
python -m aoe4_predict report

# Predict one matchup.
python -m aoe4_predict predict \
  --player-a 1270139 \
  --player-b 21150142 \
  --civ-a english \
  --civ-b delhi_sultanate \
  --map "Hill and Dale"
```

Extra feature families can be enabled on training, evaluation, civ analysis, and
tuning commands with flags such as `--add-civ-recency`, `--add-mmr-trend`,
`--add-adjusted-form`, or `--add-all-families`.

## Civilization Choice

Run the standard civilization-choice experiment:

```bash
python -m civ_choice.run --db aoe4.duckdb --seasons 10,11,12
```

Useful options:

- `--rebuild`: rebuild DuckDB civ-choice tables.
- `--no-lgbm`: skip LightGBM training.
- `--no-shap`: skip SHAP computation.

Anonymous-opponent and live-assistant entrypoints are also exposed:

```bash
civ-choice-anonymous-opponent --help
civ-choice-live --help
```

## Rating Delta Analysis

Run the rating update investigation:

```bash
python -m ratings_delta.run --db aoe4.duckdb --seasons 10,11,12
```

This builds a dataset from observed visible rating changes, fits Elo-like
baselines, optionally trains LightGBM residual models, and writes the generated
rating update report.

## Realtime Replay Snapshot Prediction

`realtime_outcome_prediction` works from parsed replay timelines and engineered
snapshots.

```bash
# Inspect parsed replay timelines.
python -m realtime_outcome_prediction inspect --limit 5

# Hydrate replay outcomes into DuckDB.
python -m realtime_outcome_prediction hydrate-outcomes --limit 500

# Build snapshot features.
python -m realtime_outcome_prediction build-dataset --limit 3000

# Train and evaluate snapshot models.
python -m realtime_outcome_prediction train
python -m realtime_outcome_prediction evaluate
```

See `src/realtime_outcome_prediction/FEATURES.md` for the replay feature
inventory.

## Backend API

The FastAPI app serves read-only historic outcome predictions.

Required local artifacts:

- `aoe4.duckdb`
- `models/aoe4_predict/lgbm_s10s11s12.txt`
- `models/aoe4_predict/lgbm_s10s11s12_meta.json`

Run locally:

```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /health`
- `GET /metadata`
- `GET /model-info`
- `POST /predict/outcome`

Example request:

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
```

Configuration is environment-variable based. See `backend/README.md` for the
full list of API settings.

## Reports

Report outputs are consolidated under `reports/`:

- `reports/generated/`: generated markdown, JSON, and CSV outputs.
- `reports/figures/`: charts referenced by generated reports.
- `reports/scripts/`: report-specific runners and analysis helpers.

Run the main report through:

```bash
python -m aoe4_predict report
```

Ad hoc report scripts can be run from the repo root, for example:

```bash
python scripts/reports/run_report_only.py
```

## Tests

Run the current test suite with:

```bash
pytest
```

The existing tests cover backend request behavior and cold-start priors. Many
training and report workflows depend on the local DuckDB and model artifacts, so
they are better treated as integration runs.

## Repository Layout

```text
backend/        FastAPI service for outcome prediction
data/           Local raw, parsed, cached, and feature data
docs/           Notes and design documentation
metadata/       Tracked static map and patch reference data
models/         Local trained model artifacts
reports/        Generated reports, figures, and report scripts
scripts/        Data, experiment, and report runners
src/            Installable Python packages
tests/          Pytest tests
```

Local generated artifacts such as `data/`, `models/`, `reports/generated/`,
`reports/figures/`, `lightning_logs/`, `*.duckdb`, and Python cache directories
should stay out of normal source-control changes unless a specific artifact is
being intentionally published.
