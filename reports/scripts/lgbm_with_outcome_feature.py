"""Check whether the match-outcome model's win probability is a useful feature
for predicting rating deltas.

Approach:
  1. For every game in the ratings-delta train/valid/test splits, load the 127
     features used by the S10+S11 outcome LightGBM model (AUC=0.71).
  2. Score with that model → P(lower-profile-id player wins) per game.
  3. Translate to per-participant outcome_win_prob:
       profile_id == profile_id_a  →  prob
       profile_id == profile_id_b  →  1 − prob
  4. Train two LightGBM rating-delta regressors:
       GBT-raw      — 11 raw CSV features (MMR, games, ratings, result)
       GBT+outcome  — same 11 features + outcome_win_prob
  5. Report test MAE comparison and feature importances.

If outcome_win_prob dominates GBT+outcome importances, the outcome model
compresses civ/form/H2H signal efficiently enough to explain Elo residuals.

Usage (from repo root):
    PYTHONPATH=src python reports/scripts/lgbm_with_outcome_feature.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aoe4_predict.config import DB_PATH, MODEL_PATH, MODEL_META_PATH
from aoe4_predict.features import _add_derived_features
from aoe4_predict.features_extra import (
    _EXT_SELECT_A, _EXT_SELECT_B,
    _p1_derived, _p2_derived, _p3_derived, _p4_derived,
    _p5_derived, _p8_derived, _p9_derived,
)
from aoe4_predict.model import CATEGORICAL_FEATURES

DATA_DIR  = REPO / "reports" / "generated"
FIG_DIR   = REPO / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── ratings-delta raw feature set (per-participant perspective) ───────────────

RD_RAW_FEATURES = [
    "result",
    "player_mmr_before", "opponent_mmr_before", "hidden_mmr_gap",
    "player_rating_before", "opponent_rating_before", "visible_rating_gap",
    "games_this_season_before", "opponent_games_this_season_before",
    "missing_player_mmr", "missing_opponent_mmr",
]

TARGET = "observed_rating_delta"


# ── load ratings-delta splits ────────────────────────────────────────────────

def _load_rd(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for col in [TARGET] + RD_RAW_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["missing_player_mmr"]   = df["player_mmr_before"].isna().astype(float)
    df["missing_opponent_mmr"] = df["opponent_mmr_before"].isna().astype(float)
    return df


def load_splits():
    print("Loading ratings-delta splits...")
    tr = _load_rd(DATA_DIR / "ratings_delta_train.csv")
    va = _load_rd(DATA_DIR / "ratings_delta_valid.csv")
    te = _load_rd(DATA_DIR / "ratings_delta_test.csv")
    for name, df in [("train", tr), ("valid", va), ("test", te)]:
        print(f"  {name:5s}: {len(df):>10,} rows, {df['game_id'].nunique():>9,} unique games")
    return tr, va, te


# ── load outcome features from DuckDB ────────────────────────────────────────

def fetch_outcome_features(game_ids: np.ndarray) -> pd.DataFrame:
    """Build game-level features for the given game_ids from source tables.

    Constructs training_features on-the-fly (filtering player_stats to S10/S11
    and the requested game_ids) then joins with player_stats_ext and h2h_priors.
    Does not modify any DuckDB table.
    """
    import duckdb

    print(f"  Querying DuckDB for {len(game_ids):,} game_ids...")
    con = duckdb.connect(str(DB_PATH))
    gids_df = pd.DataFrame({"game_id": game_ids})
    con.register("_rd_gids", gids_df)

    sql = f"""
    WITH pairs AS (
        SELECT
            a.game_id, a.started_at, a.map, a.patch, a.season,
            a.result                        AS target,
            a.profile_id                    AS profile_id_a,
            a.civ                           AS civ_a,
            a.civilization_randomized       AS civ_rand_a,
            a.mmr                           AS mmr_a,
            a.rating                        AS rating_a,
            a.games_lifetime_before         AS games_lifetime_a,
            a.wins_lifetime_before          AS wins_lifetime_a,
            a.games_season_before           AS games_season_a,
            a.wins_season_before            AS wins_season_a,
            a.days_since_last_game          AS days_since_a,
            a.civ_games_before              AS civ_games_a,
            a.civ_wins_before               AS civ_wins_a,
            a.map_games_before              AS map_games_a,
            a.map_wins_before               AS map_wins_a,
            b.profile_id                    AS profile_id_b,
            b.civ                           AS civ_b,
            b.civilization_randomized       AS civ_rand_b,
            b.mmr                           AS mmr_b,
            b.rating                        AS rating_b,
            b.games_lifetime_before         AS games_lifetime_b,
            b.wins_lifetime_before          AS wins_lifetime_b,
            b.games_season_before           AS games_season_b,
            b.wins_season_before            AS wins_season_b,
            b.days_since_last_game          AS days_since_b,
            b.civ_games_before              AS civ_games_b,
            b.civ_wins_before               AS civ_wins_b,
            b.map_games_before              AS map_games_b,
            b.map_wins_before               AS map_wins_b
        FROM player_stats a
        JOIN player_stats b
            ON a.game_id = b.game_id AND a.profile_id < b.profile_id
        INNER JOIN _rd_gids ON a.game_id = _rd_gids.game_id
        WHERE a.season IN (10, 11)
    ),
    tf AS (
        SELECT p.*,
            mp.prior_games AS prior_matchup_games,
            mp.prior_wins  AS prior_matchup_wins
        FROM pairs p
        LEFT JOIN civ_matchup_priors mp
            ON p.civ_a = mp.civ_a AND p.civ_b = mp.civ_b AND p.season = mp.season
    ),
    pse_f AS (
        SELECT pse.*
        FROM player_stats_ext pse
        INNER JOIN _rd_gids USING (game_id)
    )
    SELECT
        tf.*,
        {_EXT_SELECT_A},
        {_EXT_SELECT_B},
        h.h2h_games_before   AS h2h_games,
        h.h2h_wins_lo_before AS h2h_wins_a
    FROM tf
    LEFT JOIN pse_f ea ON tf.game_id = ea.game_id AND tf.profile_id_a = ea.profile_id
    LEFT JOIN pse_f eb ON tf.game_id = eb.game_id AND tf.profile_id_b = eb.profile_id
    LEFT JOIN h2h_priors h ON tf.game_id = h.game_id
    """

    raw = con.execute(sql).df()
    con.close()
    print(f"  DuckDB returned {len(raw):,} rows")

    # Apply the same feature derivation pipeline used at training time
    raw = _add_derived_features(raw)
    for fn in (_p1_derived, _p2_derived, _p3_derived, _p4_derived,
               _p5_derived, _p8_derived, _p9_derived):
        raw = fn(raw)

    return raw


# ── score with outcome model ──────────────────────────────────────────────────

def score_outcome(feat_df: pd.DataFrame, model: lgb.Booster,
                  feature_cols: list[str]) -> np.ndarray:
    """Return P(player_a wins) for each row in feat_df."""
    available = [c for c in feature_cols if c in feat_df.columns]
    X = feat_df[available].copy()
    for c in CATEGORICAL_FEATURES:
        if c in X.columns:
            X[c] = X[c].astype("category")
    for c in X.select_dtypes(include="object").columns:
        if c not in CATEGORICAL_FEATURES:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return model.predict(X)


# ── merge outcome prob into ratings-delta splits ──────────────────────────────

def add_outcome_prob(rd_df: pd.DataFrame, probs_df: pd.DataFrame) -> pd.DataFrame:
    """probs_df has columns: game_id, profile_id_a, p_a_wins.
    For each rd row: prob = p_a_wins if participant == player_a else 1 − p_a_wins."""
    merged = rd_df.merge(probs_df, on="game_id", how="left")
    merged["outcome_win_prob"] = np.where(
        merged["profile_id"] == merged["profile_id_a"],
        merged["p_a_wins"],
        1.0 - merged["p_a_wins"],
    )
    return merged.drop(columns=["profile_id_a", "p_a_wins"])


# ── LGB training (ratings-delta regression) ──────────────────────────────────

LGB_PARAMS = dict(
    objective        = "regression",
    metric           = ["rmse", "mae"],
    num_leaves       = 63,
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq     = 5,
    learning_rate    = 0.05,
    min_child_samples= 50,
    verbose          = -1,
    random_state     = 42,
)


def train_lgb(X_tr, y_tr, X_va, y_va, label: str,
              n_estimators: int = 800) -> lgb.LGBMRegressor:
    print(f"\n  [{label}] training on {len(X_tr):,} rows, {X_tr.shape[1]} features...")
    m = lgb.LGBMRegressor(n_estimators=n_estimators, **LGB_PARAMS)
    m.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_names=["valid"],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(200),
        ],
    )
    pred = m.predict(X_va)
    mae  = float(np.nanmean(np.abs(y_va - pred)))
    bias = float(np.nanmean(y_va - pred))
    print(f"  [{label}] valid MAE={mae:.4f}  bias={bias:+.4f}  trees={m.best_iteration_}")
    return m


def eval_metrics(model, X, y):
    pred = model.predict(X)
    v = ~np.isnan(y)
    r = y[v] - pred[v]
    return dict(
        mae  = float(np.mean(np.abs(r))),
        rmse = float(np.sqrt(np.mean(r**2))),
        bias = float(np.mean(r)),
        n    = int(v.sum()),
    )


# ── feature importance plot ───────────────────────────────────────────────────

def plot_importances(models: dict, feature_names_map: dict, fname: str):
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, (label, m) in zip(axes, models.items()):
        feats = feature_names_map[label]
        imp   = m.feature_importances_
        order = np.argsort(imp)[::-1][:15]
        ax.barh([feats[i] for i in order[::-1]], imp[order[::-1]],
                color="steelblue", alpha=0.8)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Importance (gain)")
    fig.suptitle("Ratings-delta LGB feature importances", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / fname
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import json

    # Load outcome model
    print("Loading outcome model...")
    outcome_model = lgb.Booster(model_file=str(MODEL_PATH))
    meta = json.loads(MODEL_META_PATH.read_text())
    outcome_features = meta["feature_cols"]
    print(f"  {len(outcome_features)} outcome features, {meta['n_trees']} trees")
    print(f"  Outcome model test AUC: {meta['metrics'].get('test_auc', meta['metrics'])}")

    # Load ratings-delta splits
    train, valid, test = load_splits()

    # Collect all game_ids across all splits
    all_game_ids = np.unique(np.concatenate([
        train["game_id"].values,
        valid["game_id"].values,
        test["game_id"].values,
    ]))
    print(f"\nTotal unique games to score: {len(all_game_ids):,}")

    # Fetch outcome features from DuckDB
    print("\nFetching outcome model features from DuckDB...")
    feat_df = fetch_outcome_features(all_game_ids)
    print(f"  Feature matrix: {feat_df.shape}")

    # Score with outcome model
    print("\nScoring with outcome model...")
    p_a_wins = score_outcome(feat_df, outcome_model, outcome_features)
    print(f"  Outcome prob range: [{p_a_wins.min():.3f}, {p_a_wins.max():.3f}]  "
          f"mean={p_a_wins.mean():.3f}")

    # Build game-level prob lookup (game_id, profile_id_a, p_a_wins)
    probs_df = pd.DataFrame({
        "game_id":      feat_df["game_id"].values,
        "profile_id_a": feat_df["profile_id_a"].values,
        "p_a_wins":     p_a_wins,
    })

    # Verify: overall accuracy
    target_col = "target"  # result for player_a
    if target_col in feat_df.columns:
        acc = float(np.mean((p_a_wins > 0.5) == feat_df[target_col].values))
        print(f"  Outcome model accuracy (>0.5 threshold): {acc:.4f}")

    # Merge outcome_win_prob into each split
    print("\nMerging outcome probs into splits...")
    train = add_outcome_prob(train, probs_df)
    valid = add_outcome_prob(valid, probs_df)
    test  = add_outcome_prob(test,  probs_df)

    miss = test["outcome_win_prob"].isna().mean() * 100
    print(f"  Missing outcome_win_prob in test: {miss:.1f}%")
    print(f"  outcome_win_prob vs result corr: "
          f"{test['outcome_win_prob'].corr(test['result']):.4f}")

    # Build feature matrices
    def _X(df, extra=()):
        cols = [c for c in RD_RAW_FEATURES + list(extra) if c in df.columns]
        return df[cols].copy()

    y_tr = train[TARGET].values.astype(float)
    y_va = valid[TARGET].values.astype(float)
    y_te = test[TARGET].values.astype(float)

    X_raw_tr = _X(train)
    X_raw_va = _X(valid)
    X_raw_te = _X(test)
    X_out_tr = _X(train, ["outcome_win_prob"])
    X_out_va = _X(valid, ["outcome_win_prob"])
    X_out_te = _X(test,  ["outcome_win_prob"])

    feat_raw = list(X_raw_tr.columns)
    feat_out = list(X_out_tr.columns)

    # Train models
    print("\n" + "=" * 70)
    print("Training GBT-raw (raw CSV features, predict delta directly)...")
    m_raw = train_lgb(X_raw_tr, y_tr, X_raw_va, y_va, "GBT-raw")

    print("\nTraining GBT+outcome (raw + outcome_win_prob)...")
    m_out = train_lgb(X_out_tr, y_tr, X_out_va, y_va, "GBT+outcome")

    # Test metrics
    m_raw_te = eval_metrics(m_raw, X_raw_te, y_te)
    m_out_te = eval_metrics(m_out, X_out_te, y_te)

    print("\n" + "=" * 65)
    print(f"  {'Model':<30}  {'Test MAE':>10}  {'Test RMSE':>10}  {'Bias':>8}")
    print("=" * 65)
    print(f"  {'GBT-raw':<30}  {m_raw_te['mae']:>10.4f}  {m_raw_te['rmse']:>10.4f}  "
          f"{m_raw_te['bias']:>+8.4f}")
    print(f"  {'GBT+outcome_win_prob':<30}  {m_out_te['mae']:>10.4f}  "
          f"{m_out_te['rmse']:>10.4f}  {m_out_te['bias']:>+8.4f}")
    delta_mae = m_raw_te['mae'] - m_out_te['mae']
    print(f"  {'Δ MAE (raw − outcome)':<30}  {delta_mae:>+10.4f}")
    print("=" * 65)

    # Feature importances
    print(f"\nFeature importances — GBT+outcome (gain-based):")
    imp   = m_out.feature_importances_
    total = imp.sum()
    order = np.argsort(imp)[::-1]
    cumsum = 0.0
    for i in order:
        pct = imp[i] / total * 100
        cumsum += pct
        marker = " ◄ outcome model" if feat_out[i] == "outcome_win_prob" else ""
        print(f"  {feat_out[i]:<38s} {pct:>6.1f}%  (cum {cumsum:>5.1f}%){marker}")
        if cumsum > 99:
            break

    print(f"\nFeature importances — GBT-raw (gain-based):")
    imp_r = m_raw.feature_importances_
    total_r = imp_r.sum()
    order_r = np.argsort(imp_r)[::-1]
    cumsum_r = 0.0
    for i in order_r:
        pct = imp_r[i] / total_r * 100
        cumsum_r += pct
        print(f"  {feat_raw[i]:<38s} {pct:>6.1f}%  (cum {cumsum_r:>5.1f}%)")
        if cumsum_r > 99:
            break

    # Plot
    plot_importances(
        {"GBT-raw": m_raw, "GBT+outcome": m_out},
        {"GBT-raw": feat_raw, "GBT+outcome": feat_out},
        "lgbm_outcome_feature_importances.png",
    )

    # Summary interpretation
    outcome_pct = imp[feat_out.index("outcome_win_prob")] / total * 100
    print(f"\n=== Summary ===")
    print(f"  outcome_win_prob importance: {outcome_pct:.1f}% of total gain")
    print(f"  MAE improvement from adding outcome_win_prob: {delta_mae:+.4f}")
    if outcome_pct > 30:
        print("  ► outcome_win_prob is a major driver — the outcome model efficiently")
        print("    summarizes civ/form/H2H interactions for delta prediction.")
        print("    Direction: use P3 residuals as next target for the outcome-augmented GBT.")
    elif outcome_pct > 10:
        print("  ► outcome_win_prob contributes meaningfully alongside raw MMR features.")
    else:
        print("  ► outcome_win_prob adds little — raw MMR already captures the signal.")

    print("\nDone.")


if __name__ == "__main__":
    main()
