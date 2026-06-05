# AOE4

Tools and experiments for predicting Age of Empires IV match outcomes from
ranked 1v1 player and match data.

The repository centers on a DuckDB match warehouse (`aoe4.duckdb`) plus feature
generation, reporting, and model-training code for outcome prediction.

## Replay Data

This repo also includes replay fetching and bookkeeping support for building a
larger real-time outcome prediction dataset. The replay harvester can label
candidate matches from the existing match database, fetch raw replay files,
track download and parse status, and separate balanced ladder samples from
top-player coverage samples.

Detailed behavior, commands, storage layout, and usage notes live in
[`replay_harvest/README.md`](replay_harvest/README.md).
