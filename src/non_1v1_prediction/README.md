# non_1v1_prediction — 4v4 matchmaking predictability

Investigates whether AOE4 ranked **4v4** matchmaking produces *predictable* matches
(prompted by Reddit complaints). Premise: **bad matchmaking = the outcome is predictable
before the match begins**. We fit a pre-game outcome model and measure how far above the
0.50 coin-flip ideal the predictability sits, and which categories of match are most decided.

## Quick start

```bash
export PYTHONPATH=src        # the env with duckdb/lightgbm/shap is `python3`

python3 -m non_1v1_prediction download --mode rm_4v4 --seasons 9,10,11   # → data/team/
python3 -m non_1v1_prediction ingest   --mode rm_4v4 --seasons 9,10,11   # → aoe4_team.duckdb
python3 -m non_1v1_prediction build-network                              # teammate graph → reports/non_1v1_teammate_network_report.md
python3 -m non_1v1_prediction report    --mode rm_4v4 --seasons 9,10,11  # → reports/non_1v1_rm_4v4_predictability_report.md
```

`train` and `evaluate` are lighter subcommands that print test-set metrics without writing
the full report.

## Design

- **Separate database** `aoe4_team.duckdb` — the 1v1 `aoe4.duckdb` is never written, only
  `ATTACH`ed **read-only** to enrich each team player with their 1v1-ladder skill (leakage-free
  ASOF join: most recent 1v1 game strictly before the team match).
- **One row per match**; Team A = team with the lower minimum `profile_id`; target = Team A won.
  Team-swap augmentation removes any A/B orientation signal.
- **Features** (all strictly pre-game): per-team MMR aggregates + A−B diffs; the **boost/carry
  exploit family** (within-team dispersion — `carry_gap`, `n_below_floor`, `n_carried` — and the
  1v1-smurf signal `n_smurf_like` / `onev1_max_minus_skill`); team-ladder history means; civ
  familiarity; map/server/patch/season context.
- **Baselines**: constant base-rate, and a one-feature **MMR-mean-diff logistic** — the crux. If
  the raw skill gap alone predicts winners, the matchmaker is handing out decided games; LightGBM's
  lift over it measures the remaining (carry / smurf / context) structure.

## Teammate network & premade features

`network.py` builds a **teammate co-occurrence graph** (edge = a pair who played ≥ `TEAMMATE_X`
games together, summed across all team modes). The threshold is validated against an
**opposite-team random baseline** — two players can't queue to be *against* each other, so
opposite-team co-occurrence is pure chance; at x=5 same-team pairs are ~7× the random rate.

Premade status enters the outcome model as **leakage-free weekly snapshots**: a pair counts as
premade for a match only if their cumulative co-team count crossed `x` in an *earlier* ISO week
(`establish_week < match_week`). Team features: `n_premade_pairs`, `max/mean_prior_coteam`,
`team_is_premade`, `premade_partners_mean`, plus match-level `both_premade` / `premade_xor`. The
full-window graph (`teammate_edges`, `player_network_stats` — degree/clustering/components) is
**descriptive only and never fed to the model.**

The per-mode report adds a premade section: prevalence, and premade-vs-solo win rate controlling
for MMR (in balanced 4v4 games the premade side wins ~54% — a real coordination edge beyond
skill, though the AUC lift is small).

## Notes

- Dump URLs are signed GCS links that **expire**, so `download.py` scrapes them fresh each run.
- Code is parameterized by `--mode`, so `rm_2v2` / `rm_3v3` can be added for comparison later.
- aoe4world dumps do not expose queue grouping, so premade-party detection is approximated via
  the carry/dispersion features rather than observed directly.
