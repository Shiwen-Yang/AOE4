# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AOE4 predicts Age of Empires IV ranked 1v1 match outcomes given two player profile IDs. The prototype uses S10+S11 data (~2.5M games) with temporal train/validate/test splits. A DuckDB database (`aoe4.duckdb`) stores parsed match data; LightGBM is the primary model.

## CLI commands

All commands must be run with `src/` on the Python path (packages live under `src/`):

```bash
export PYTHONPATH=src   # or prefix each command with PYTHONPATH=src

# One-time setup: ingest seasons (default: all found in data/; --seasons 10,11 for prototype)
python -m aoe4_predict ingest --seasons 10,11          # skips already-ingested games
python -m aoe4_predict ingest --seasons 10,11 --force  # re-ingest everything

# Load map/patch metadata CSVs into DuckDB (run before training P6/P7 families)
python -m aoe4_predict ingest-metadata --metadata-dir metadata/

# Print data quality report (saves to data_quality_report.json)
python -m aoe4_predict quality

# Build features and train LightGBM (P6/P7 disabled by default in --add-all-families)
python -m aoe4_predict train --seasons 10,11
python -m aoe4_predict train --add-all-families                 # P1-P5,P8,P9
python -m aoe4_predict train --add-all-families --also-train-xgb  # also train XGBoost

# Hyperparameter tuning via Optuna (50 trials default; re-trains final model with best params)
python -m aoe4_predict tune --model lgbm --n-trials 50 --add-all-families
python -m aoe4_predict tune --model xgb  --n-trials 50 --add-all-families
python -m aoe4_predict tune --model lgbm --n-trials 50 --timeout 3600  # wall-clock limit (seconds)
python -m aoe4_predict tune --model lgbm --n-trials 50 --no-retrain    # search only, skip final fit

# Compare model against baselines on the temporal test set
python -m aoe4_predict evaluate

# Skill-stratified civ familiarity analysis → reports/civ_familiarity_report.md
python -m aoe4_predict analyze-civ --add-all-families

# Generate full analysis report with SHAP and ablation → reports/analysis_report.md
python -m aoe4_predict report

# Report-only helper scripts live under reports/scripts/
python reports/scripts/run_report_only.py

# Predict a match (civ and map are optional)
python -m aoe4_predict predict --player-a 3507399 --player-b 6914972
python -m aoe4_predict predict --player-a 3507399 --player-b 6914972 --map "Dry Arabia"
python -m aoe4_predict predict --player-a 3507399 --player-b 6914972 --civ-a english --civ-b french --map "Dry Arabia"
```

All commands accept `--db <path>` to override the default `aoe4.duckdb` location.

### Feature family flags

These flags are accepted by `train`, `tune`, `evaluate`, and `analyze-civ`. They can be combined individually or use `--add-all-families` (which enables P1–P5, P8–P9 but **not** P6/P7):

```
--add-civ-recency         P1: time-windowed civ history (7/30/60-day windows)
--add-mmr-trend           P2: MMR volatility, slope over last 3/5/10/20 games
--add-adjusted-form       P3: recent win-rate over last 5/10/20 games
--add-duration-profile    P4: short/long game split stats
--add-head-to-head        P5: cumulative head-to-head record between two players
--add-map-archetypes      P6: map taxonomy features (opt-in; disabled in --add-all-families)
--add-patch-priors        P7: patch-age and patch-type features (opt-in; disabled in --add-all-families)
--add-low-history-detail  P8: cold-start flags for new/low-history players
--add-activity-session    P9: time-windowed activity (7/14/30/60-day game counts)
```

## Dependencies

No `requirements.txt` or `pyproject.toml` exists. Key packages:

- **Data**: `duckdb`, `pandas`, `numpy`
- **Models**: `lightgbm`, `xgboost`
- **Tuning**: `optuna`
- **Analysis**: `shap`, `matplotlib`

## Package structure

Source packages live under `src/`. Run everything with `PYTHONPATH=src` from the repo root.

```
src/
  aoe4_predict/
    config.py        — paths, constants (PRIOR_STRENGTH, NEW_PLAYER_THRESHOLD, season lists)
    db.py            — DuckDB connection, DDL (games + participants tables), init_schema(),
                       ingest_metadata() for map/patch CSV loading
    ingest.py        — JSON.gz → DuckDB; handles S3 missing mmr/input_type; skip_existing=True
    data_quality.py  — quality report: missingness, civ coverage, MMR continuity across seasons
    features.py      — player_stats (window functions), civ_matchup_priors, training_features tables;
                       get_inference_features() for prediction time
    features_extra.py — extended feature families P1-P9 (P6/P7 disabled by default);
                        DISABLED_FAMILIES, FAMILY_FEATURES, extend_training_features()
    baselines.py     — ConstantBaseline, MMRLogisticBaseline, CivMapBucketBaseline
    model.py         — LightGBM + XGBoost training/save/load; _temporal_split(),
                       train_xgb(), _predict_xgb(), load_xgb()
    tune.py          — Optuna TPE hyperparameter search for LightGBM and XGBoost;
                       700K-row trial subsampling; saves best_params.json
    civ_analysis.py  — Skill-stratified civ familiarity analysis; player_civ_extra SQL table;
                       5 avg-MMR buckets × 4-step ablation; SHAP by bucket; civ_familiarity_report.md
    evaluate.py      — AUC/LogLoss/Brier/calibration, subgroup breakdown, compare_baselines()
    predict.py       — predict_match() → structured dict with warnings
    output.py        — format_prediction() → terminal report string
    report.py        — analysis_report.md: leakage audit, SHAP, ablation, XGBoost comparison
    cli.py           — argparse subcommands: ingest, ingest-metadata, quality, train, tune,
                       analyze-civ, evaluate, report, predict
  civ_choice/        — predicts which civilization a player will pick
  ratings_delta/     — models rating-point changes after a match
```

## Architecture

### Data layer
- **Schema**: `games` (one row per match) + `participants` (one row per player per match, normalized). Player A/B ordering uses `profile_id < profile_id` convention — arbitrary but consistent.
- **Season differences**: S3 has no `mmr`, `mmr_diff`, or `input_type` — ingest inserts NULLs. S4+ has all fields.
- **MMR**: confirmed no hard reset between seasons (median cross-season MMR jump = 15 points). MMR is the primary skill signal; visible rating is secondary with explicit `missing_mmr_*` indicators.

### Feature computation
Two materialized tables built in `features.py`:

1. **`player_stats`** — one row per participant per game with leakage-free window functions:
   - `games_lifetime_before`, `wins_lifetime_before` — cumulative counts using `ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING`
   - `civ_games_before`, `civ_wins_before` — per civ
   - `map_games_before`, `map_wins_before` — per map
   - `days_since_last_game` — via LAG

2. **`civ_matchup_priors`** — win rates for each civ pair, aggregated by season using a cumulative window over previous seasons only. Joined to training rows on `(civ_a, civ_b, season)`.

Derived features (computed in Python, not SQL): smoothed win rates using additive prior `(wins + 10 × 0.5) / (games + 10)`, MMR difference, skill fallback (`mmr` → `rating` → None), missing indicators, context flags (`civs_known`, `map_known`, `full_context_known`).

### Models
- **Target**: `result` of player A (lower `profile_id`), so `target=1` means the lower-ID player wins
- **Temporal split** (no random splits): train → valid → test are chronological slices (70/15/15%)
- **LightGBM**: native categorical features (`civ_a`, `civ_b`, `map`, `patch`, `season` + P6 taxonomy); saved as `models/lgbm_s9s10_test_s11.txt` + `lgbm_s9s10_test_s11_meta.json`
- **XGBoost**: same feature set, `enable_categorical=True`; saved as `models/xgb_s9s10_test_s11.ubj` + `xgb_s9s10_test_s11_meta.json`
- **Hyperparameter tuning**: Optuna TPE, 50 trials per model, 700K-row subsampled training for each trial; best params saved to `models/{lgbm,xgb}_best_params.json`
- **Current results on test set** (S10+S11, 196 features, tuned params):
  - Constant 0.5: AUC 0.500
  - MMR logistic: AUC 0.615
  - Civ/map bucket: AUC 0.541
  - **LightGBM (tuned): AUC 0.7114**, Brier 0.2150
  - **XGBoost (default): AUC 0.7111**, Brier 0.2151 (tuning in progress)
- **P6/P7 families** (map archetypes, patch priors): implemented but disabled in `--add-all-families` — showed Brier Δ < 0.001 on S10+S11; manually re-enable with `--add-map-archetypes` / `--add-patch-priors`

### Inference
`predict_match(player_a_id, player_b_id, civ_a, civ_b, map_name)` queries the DB for current player stats, constructs the feature dict, and scores with the saved model. Returns context level ("id_only" / "map_known" / "civ_known" / "full_context") and reliability warnings for sparse data.

## Data notes

- Data lives in `data/games_rm_1v1_s{3..11}.json.gz` (9 seasons, ~11.3M total games)
- The `.json.gz` files are the raw backup; DuckDB stores only parsed columnar data
- `kind = rm_1v1` for all records in these dump files (`rm_solo` is an API variant, not present in dumps)
- 18 civilizations in S10+S11 (original 10 + 8 DLC); civ matchup priors handle new civs automatically
- MMR missing: ~6% of participants in S10+S11; rating missing: ~12%
- `aoe_analysis.ipynb` at the repo root contains exploratory analysis
- Report artifacts are consolidated under `reports/`: markdown/JSON outputs at the top level, charts under `reports/figures/`, and ad hoc report runners under `reports/scripts/`

## Known limitations

- Training uses only S10+S11 — player history features reflect only those seasons, not the player's full career
- Civ matchup priors use season-level granularity (not patch-level) due to computational cost
- DuckDB uses exclusive file lock — only one process can hold a writable connection at a time; run training/tuning/analysis commands sequentially
- LightGBM tuned model hits 1000-round limit — lower LR (0.029) means it could potentially improve with more rounds
- No automated test suite — correctness is verified via `quality`, `evaluate`, and `report` CLI commands
