"""Compare three LightGBM approaches for rating-delta prediction.

  GBT-raw      — LGB on raw features only (same inputs P3 uses)
  GBT-residual — LGB trained to predict P3 residuals (traditional stacking)
  GBT+P3       — LGB on raw features + p3_pred as additional feature

If P3_pred dominates the importance ranking in GBT+P3, the formula captures the
complex patterns efficiently.  Any raw features that remain important alongside
p3_pred are candidates for formula extension.

Usage (from repo root):
    PYTHONPATH=src python reports/scripts/lgbm_with_p3_feature.py
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

from ratings_delta.parametric import P3Model

DATA_DIR  = REPO / "reports" / "generated"
FIG_DIR   = REPO / "reports" / "figures"
MODEL_DIR = REPO / "models"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── data loading ──────────────────────────────────────────────────────────────

NUMERIC_COLS = [
    "result",
    "player_mmr_before", "opponent_mmr_before", "hidden_mmr_gap",
    "player_rating_before", "opponent_rating_before", "visible_rating_gap",
    "games_this_season_before", "opponent_games_this_season_before",
]

def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for col in NUMERIC_COLS + ["observed_rating_delta"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Missing-MMR indicators
    df["missing_player_mmr"]    = df["player_mmr_before"].isna().astype(float)
    df["missing_opponent_mmr"]  = df["opponent_mmr_before"].isna().astype(float)
    return df


def load_splits():
    print("Loading splits...")
    tr = _load(DATA_DIR / "ratings_delta_train.csv")
    va = _load(DATA_DIR / "ratings_delta_valid.csv")
    te = _load(DATA_DIR / "ratings_delta_test.csv")
    for name, df in [("train", tr), ("valid", va), ("test", te)]:
        print(f"  {name:5s}: {len(df):>10,} rows")
    return tr, va, te


# ── feature sets ──────────────────────────────────────────────────────────────

RAW_FEATURES = NUMERIC_COLS + ["missing_player_mmr", "missing_opponent_mmr"]

def build_X(df: pd.DataFrame, extra_cols: list[str] = ()) -> pd.DataFrame:
    cols = RAW_FEATURES + list(extra_cols)
    return df[[c for c in cols if c in df.columns]].copy()


# ── LGB training ──────────────────────────────────────────────────────────────

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

def train_lgb(
    X_tr, y_tr, X_va, y_va,
    label: str,
    n_estimators: int = 800,
) -> lgb.LGBMRegressor:
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
    pred_va = m.predict(X_va)
    mae  = float(np.nanmean(np.abs(y_va - pred_va)))
    bias = float(np.nanmean(y_va - pred_va))
    print(f"  [{label}] valid MAE={mae:.4f}  bias={bias:+.4f}  trees={m.best_iteration_}")
    return m


def eval_metrics(model, X, y, offset=None):
    """Evaluate model; optionally add a fixed offset (for GBT-residual)."""
    pred = model.predict(X)
    if offset is not None:
        pred = pred + offset
    v = ~np.isnan(y)
    r = y[v] - pred[v]
    return dict(
        mae   = float(np.mean(np.abs(r))),
        rmse  = float(np.sqrt(np.mean(r ** 2))),
        bias  = float(np.mean(r)),
        n     = int(v.sum()),
    )


# ── feature importance plot ───────────────────────────────────────────────────

def plot_importances(models: dict[str, lgb.LGBMRegressor],
                     feature_names_map: dict[str, list[str]],
                     fname: str):
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, (label, m) in zip(axes, models.items()):
        feats = feature_names_map[label]
        imp   = m.feature_importances_
        order = np.argsort(imp)[::-1][:20]
        ax.barh(
            [feats[i] for i in order[::-1]],
            imp[order[::-1]],
            color="steelblue", alpha=0.8,
        )
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Importance (gain)")
    fig.suptitle("LGB feature importances", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / fname
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    train, valid, test = load_splits()

    # ── fit P3 (two-stage: D on 1/6, rest on full) ───────────────────────────
    print("\nFitting P3 model (two-stage)...")
    p3 = P3Model()
    train_sub = train.sample(frac=1/6, random_state=42).reset_index(drop=True)
    p3.fit_two_stage(train_sub, train)

    for split_df, name in [(train, "train"), (valid, "valid"), (test, "test")]:
        split_df["p3_pred"] = p3.predict(split_df)
    print(f"  P3 train MAE: {np.nanmean(np.abs(train['observed_rating_delta'] - train['p3_pred'])):.4f}")
    print(f"  P3 test  MAE: {np.nanmean(np.abs(test['observed_rating_delta']  - test['p3_pred'])):.4f}")

    # ── targets ───────────────────────────────────────────────────────────────
    y_tr  = train["observed_rating_delta"].values.astype(float)
    y_va  = valid["observed_rating_delta"].values.astype(float)
    y_te  = test["observed_rating_delta"].values.astype(float)
    r_tr  = (train["observed_rating_delta"] - train["p3_pred"]).values.astype(float)
    r_va  = (valid["observed_rating_delta"] - valid["p3_pred"]).values.astype(float)

    # ── build feature matrices ────────────────────────────────────────────────
    X_raw_tr  = build_X(train)
    X_raw_va  = build_X(valid)
    X_raw_te  = build_X(test)

    X_p3_tr   = build_X(train, ["p3_pred"])
    X_p3_va   = build_X(valid, ["p3_pred"])
    X_p3_te   = build_X(test,  ["p3_pred"])

    feat_raw = list(X_raw_tr.columns)
    feat_p3  = list(X_p3_tr.columns)

    # ── train three models ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Training GBT-raw (raw features, predict delta directly)...")
    m_raw = train_lgb(X_raw_tr, y_tr, X_raw_va, y_va, "GBT-raw")

    print("\nTraining GBT-residual (raw features, predict P3 residual)...")
    m_resid = train_lgb(X_raw_tr, r_tr, X_raw_va, r_va, "GBT-residual")

    print("\nTraining GBT+P3 (raw features + p3_pred, predict delta directly)...")
    m_p3 = train_lgb(X_p3_tr, y_tr, X_p3_va, y_va, "GBT+P3")

    # ── test evaluation ───────────────────────────────────────────────────────
    p3_pred_te = test["p3_pred"].values.astype(float)

    m_raw_te   = eval_metrics(m_raw,   X_raw_te, y_te)
    m_resid_te = eval_metrics(m_resid, X_raw_te, y_te, offset=p3_pred_te)
    m_p3_te    = eval_metrics(m_p3,    X_p3_te,  y_te)

    # P3 alone (formula, no GBT)
    p3_v = ~np.isnan(y_te) & ~np.isnan(p3_pred_te)
    p3_mae  = float(np.mean(np.abs(y_te[p3_v] - p3_pred_te[p3_v])))
    p3_bias = float(np.mean(y_te[p3_v] - p3_pred_te[p3_v]))

    print("\n" + "=" * 75)
    print(f"  {'Model':<32}  {'Test MAE':>10}  {'Test RMSE':>10}  {'Bias':>8}")
    print("=" * 75)
    print(f"  {'P3 (formula only)':<32}  {p3_mae:>10.4f}  {'—':>10}  {p3_bias:>+8.4f}")
    print(f"  {'GBT-raw':<32}  {m_raw_te['mae']:>10.4f}  {m_raw_te['rmse']:>10.4f}  {m_raw_te['bias']:>+8.4f}")
    print(f"  {'P3 + GBT-residual':<32}  {m_resid_te['mae']:>10.4f}  {m_resid_te['rmse']:>10.4f}  {m_resid_te['bias']:>+8.4f}")
    print(f"  {'GBT+P3 (p3_pred as feature)':<32}  {m_p3_te['mae']:>10.4f}  {m_p3_te['rmse']:>10.4f}  {m_p3_te['bias']:>+8.4f}")
    print("=" * 75)

    # ── feature importances for GBT+P3 ───────────────────────────────────────
    print(f"\nFeature importances — GBT+P3 (gain-based):")
    imp   = m_p3.feature_importances_
    total = imp.sum()
    order = np.argsort(imp)[::-1]
    cumsum = 0.0
    for i in order:
        pct = imp[i] / total * 100
        cumsum += pct
        marker = " ◄ P3 prediction" if feat_p3[i] == "p3_pred" else ""
        print(f"  {feat_p3[i]:<38s} {pct:>6.1f}%  (cum {cumsum:>5.1f}%){marker}")
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

    # ── plot ──────────────────────────────────────────────────────────────────
    plot_importances(
        {"GBT-raw": m_raw, "GBT+P3": m_p3, "GBT-residual": m_resid},
        {"GBT-raw": feat_raw, "GBT+P3": feat_p3, "GBT-residual": feat_raw},
        "lgbm_p3_feature_importances.png",
    )

    # ── GBT+P3 importances excluding p3_pred (what raw features still matter?) ─
    print(f"\nGBT+P3 importances excluding p3_pred (residual signal):")
    imp_ex    = {feat_p3[i]: imp[i] for i in range(len(feat_p3)) if feat_p3[i] != "p3_pred"}
    total_ex  = sum(imp_ex.values())
    if total_ex > 0:
        for feat, val in sorted(imp_ex.items(), key=lambda x: -x[1]):
            print(f"  {feat:<38s} {val/total_ex*100:>6.1f}% of non-p3 importance")

    print("\nDone.")


if __name__ == "__main__":
    main()
