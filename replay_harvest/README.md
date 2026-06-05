# Replay Harvest

Tools for acquiring raw Age of Empires IV replay files against the existing
`aoe4.duckdb` match warehouse.

Initialize tables:

```bash
python -m replay_harvest init-schema
```

Label the balanced RM 1v1 sample:

```bash
python -m replay_harvest label-balanced --limit 10000
```

Label complete coverage for the current top 100 canonical AoE4World players:

```bash
python -m replay_harvest label-top100
```

Download slowly:

```bash
python -m replay_harvest download --group balanced_10k --limit 1000 --sleep-min 15 --sleep-max 30
```

Parse downloaded files:

```bash
python -m replay_harvest parse-downloaded --group balanced_10k --limit 100
```

Raw files are written under `data/replays/raw/YYYY-MM-DD/`. DuckDB remains the
source of truth for download and parse status.
