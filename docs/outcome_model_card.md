# AOE4 Outcome Model — Card & Serving Reference

Reference for building the front end against the live outcome-prediction service.
Covers what the model is, how it's served, how well it performs, the civ-matchup
data caveat, and its limitations.

Last updated: 2026-06-13.

---

## 1. What it predicts

Given two ranked **1v1** player profile IDs (and optionally each player's civ and
the map), the model returns **P(player A wins)** — `win_prob_a` (and `win_prob_b = 1 − win_prob_a`).
"Player A" is simply whichever ID is sent as `player_a_id`.

## 2. Model setup (the deployed "API30" model)

| Property | Value |
|---|---|
| Algorithm | LightGBM (gradient-boosted trees), binary objective |
| Artifact | `models/aoe4_predict/lgbm_api30_recent_only.txt` (+ `_meta.json`) |
| Features | **131** (53 base + 5 categorical `civ_a, civ_b, map, patch, season` + P1 civ-recency 28 + P3 recent-form 15 + P4 duration-profile 28 + P5 head-to-head 3) |
| Training data | Ranked 1v1, **seasons 10 + 11 + 12** (~5.0M games) |
| Split | Temporal (chronological), 70 / 15 / 15 → train 3.48M, valid 0.75M, **test 0.75M** (most-recent slice, held out) |
| Trees / params | 999 trees, lr 0.029, num_leaves 167, L1 3.55, L2 3.11, feature_fraction 0.60 (Optuna-tuned) |
| Key design | Every history-derived feature is computed from each player's **last 30 games only** ("API30" / `recent_only`), matching exactly what one aoe4world recent-games page provides. MMR/rating are taken from the most recent game. |

**Most influential features** (LightGBM gain share): `skill_diff` 16.6%, `mmr_diff`
9.0%, `wins_lifetime_b` 8.0%, `wins_lifetime_a` 7.8%, `civ_a` 4.1%. Skill (MMR) and
recent volume dominate; civ/map context is secondary.

## 3. Serving architecture (DB-free, live)

The backend serves predictions **without a database** by calling aoe4world at request time.

- **Endpoint:** `POST /predict/outcome`
- **Request:** `{ player_a_id, player_b_id, civ_a?, civ_b?, map_name? }`
- **Response:** `prediction.{win_prob_a, win_prob_b}`, plus `data_quality`
  (`context_level`, `warnings`, `imputations`, `unseen_categories`, **`data_freshness`**)
  and `model` version labels.
- **API calls per prediction: 2** — one recent-games page per player
  (`/api/v0/players/{id}/games?mode=rm_solo`), fetched concurrently. Each page also
  carries the player's current MMR/rating, so no extra skill lookup.
- **Civ-matchup priors** come from a **cached global snapshot**
  (`/api/v0/stats/rm_solo/matchups`, refreshed ~6h), not a per-prediction call.
- Toggle `AOE4_FEATURE_SOURCE=api|db` (default `api`). `db` mode uses local DuckDB
  and the full-history model — for local testing/experiments only.

### Failure behavior (what the front end should expect)

| Situation | HTTP | Behavior |
|---|---|---|
| aoe4world timeout / 5xx (after retries) | **503** | detail names the failed player |
| Malformed aoe4world response | **502** | — |
| `before_timestamp` sent | **400** | historical scoring unsupported in api mode |
| Player unknown / 0 recent games | **200** | cold-start priors used; warning emitted (no error) |
| <10 recent games / missing MMR | **200** | prediction returned + "less reliable" warning |
| Unseen civ / map / patch / season | **200** | treated as an unknown category by the model + warning |

`data_freshness` reports per-player recent-game counts and latest game timestamps,
so the UI can show how current the inputs are.

## 4. Performance

**Held-out test set (most-recent slice of S10–S12):**

| Metric | Value |
|---|---|
| AUC | **0.6948** |
| Log loss | 0.6277 |
| Brier | 0.2198 |
| Accuracy @0.5 | 0.6359 |

Reference baselines on the same data: constant-0.5 ≈ 0.500 AUC; MMR-only logistic ≈ 0.61 AUC.

**Calibration** (key for showing probabilities, not just a winner): on a fully unseen
future season the expected calibration error (ECE) is **< 1.3%**. Probabilities are
trustworthy mid-range; the model is mildly *under-confident* on heavy favorites
(predicts ~0.84 where the favorite actually wins ~0.91).

**Cross-season generalization** (sliding 2-season train window → predict the next,
unseen season — the basis for trusting it on a brand-new season like S13):

| Train | Test (unseen) | AUC | ECE | vs in-distribution |
|---|---|---|---|---|
| S7+S8 | S9 | 0.6963 | 0.4% | −0.002 |
| S8+S9 | S10 | 0.6800 | 0.7% | +0.024 |
| S9+S10 | S11 | 0.6948 | 0.4% | +0.004 |
| S10+S11 | S12 | 0.6734 | 1.3% | +0.023 |

Out-of-distribution AUC averages **0.686** (range 0.673–0.696); the seasonal penalty
is at most ~2.4 AUC points and calibration holds. Note this is the *conservative* case
(zero games from the new season in training) — it shrinks as the new season fills in and
the model is retrained.

## 5. Civ-matchup prior: source drift

The model was trained with civ-matchup priors aggregated from local history (cumulative
over prior seasons). Live serving instead reads the aoe4world matchups snapshot (a
*current* snapshot). They are not identical. Measured drift vs the training source
(shared civ pairs with ≥50 games each):

- **Non-mirror matchups:** mean |Δ win-rate| **3.5 pp**, median 2.7 pp, p90 7.3 pp,
  games-weighted **2.8 pp**.
- **Mirror matchups (same civ both sides):** the endpoint reports a degenerate **100%**
  win rate. This is **corrected in code** — mirrors are forced to a symmetric 50%.

**Impact is small:** the three civ-matchup features account for only **~1.5% of total
model gain** (`prior_matchup_wr_a` 1.0%, the two count features ~0.2% each), and the
feature is additively smoothed. A ~3 pp drift on a ~1%-weight, smoothed feature moves
predictions negligibly. For exact reproduction of training semantics (e.g. historical
backtests) the local DuckDB prior is still required.

## 6. Limitations

- **Recency-only ceiling (~0.695 AUC).** Using only the last 30 games is what makes the
  service DB-free, but it caps accuracy. Adding full-history "career" signal (peak/career
  MMR, career win rate) lifts test AUC to ~0.705 — but that needs a stored history cache,
  not recoverable from a single aoe4world page. We deliberately traded ~1 AUC point for
  zero database dependency.
- **Two external calls per prediction.** Latency and availability depend on aoe4world;
  a 45s per-player and ~6h matchup cache mitigate bursts. Sustained outages → 503.
- **Snapshot semantics.** Civ-matchup priors are a live snapshot (optionally patch-filtered),
  not an as-of historical prior; see §5. The endpoint is also undocumented and could change.
- **New season / patch / civ.** S13 (and new DLC civs/maps) are unseen categoricals; the
  model handles them as "unknown" buckets — works, but with the ~2.4-point seasonal penalty
  in §4 until retrained on the new season.
- **Civ/map name coupling.** Assumes aoe4world civ/map slugs match the model's vocabulary
  (same data source); mismatches degrade gracefully to the unknown bucket + a warning.
- **Non-determinism.** As a player plays more games, the same request can return a slightly
  different probability over time — expected for a live recent-window model. `data_freshness`
  exposes the inputs' currency.
- **New / low-history players.** With few or no recent games the model falls back to
  cold-start skill priors; treat these predictions as low-confidence (flagged in `warnings`).
- **Scope.** Ranked **1v1 (rm_solo)** only; not validated for team games, custom games, or
  non-ranked modes.
