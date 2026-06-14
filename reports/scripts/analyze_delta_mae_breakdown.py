"""MAE breakdown of P3 (formula) vs GBT-raw by games played and MMR tier.

Shows where each model loses accuracy: placement vs established players, and
across the skill spectrum.  Helps identify which buckets most benefit from
a GBT over the parametric formula.

Usage (from repo root):
    PYTHONPATH=src python reports/scripts/analyze_delta_mae_breakdown.py
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

def _load(path):
    df = pd.read_csv(path, low_memory=False)
    for col in NUMERIC_COLS + ["observed_rating_delta"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["missing_player_mmr"]   = df["player_mmr_before"].isna().astype(float)
    df["missing_opponent_mmr"] = df["opponent_mmr_before"].isna().astype(float)
    return df


def load_splits():
    print("Loading splits...")
    tr = _load(DATA_DIR / "ratings_delta_train.csv")
    va = _load(DATA_DIR / "ratings_delta_valid.csv")
    te = _load(DATA_DIR / "ratings_delta_test.csv")
    for name, df in [("train", tr), ("valid", va), ("test", te)]:
        print(f"  {name:5s}: {len(df):>10,} rows")
    return tr, va, te


# ── GBT-raw training ──────────────────────────────────────────────────────────

RAW_FEATURES = NUMERIC_COLS + ["missing_player_mmr", "missing_opponent_mmr"]

LGB_PARAMS = dict(
    objective="regression", metric=["rmse", "mae"],
    num_leaves=63, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=5,
    learning_rate=0.05, min_child_samples=50,
    verbose=-1, random_state=42,
)


def train_gbt_raw(train, valid):
    X_tr = train[[c for c in RAW_FEATURES if c in train.columns]]
    y_tr = train["observed_rating_delta"].values.astype(float)
    X_va = valid[[c for c in RAW_FEATURES if c in valid.columns]]
    y_va = valid["observed_rating_delta"].values.astype(float)

    print("  Training GBT-raw...")
    m = lgb.LGBMRegressor(n_estimators=800, **LGB_PARAMS)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_names=["valid"],
          callbacks=[lgb.early_stopping(50, verbose=False),
                     lgb.log_evaluation(200)])
    pred_va = m.predict(X_va)
    mae = float(np.nanmean(np.abs(y_va - pred_va)))
    print(f"  GBT-raw valid MAE={mae:.4f}  trees={m.best_iteration_}")
    return m


# ── bucket definitions ────────────────────────────────────────────────────────

GAMES_BINS   = [0, 10, 50, 100, 200, np.inf]
GAMES_LABELS = ["0–9\n(placement)", "10–49\n(settling)",
                "50–99", "100–199", "200+"]

MMR_BINS   = [0, 800, 1000, 1200, 1400, 1600, np.inf]
MMR_LABELS = ["<800\n(Bronze)", "800–1000\n(Silver)", "1000–1200\n(Gold)",
              "1200–1400\n(Plat)", "1400–1600\n(Diamond)", "1600+\n(Conq)"]


def assign_games_bucket(df):
    return pd.cut(
        df["games_this_season_before"].clip(lower=0),
        bins=GAMES_BINS, labels=GAMES_LABELS, right=False,
    )


def assign_mmr_bucket(df):
    # Use player_mmr_before; fall back to player_rating_before if missing
    mmr = df["player_mmr_before"].where(
        df["player_mmr_before"].notna(), df["player_rating_before"]
    )
    return pd.cut(mmr, bins=MMR_BINS, labels=MMR_LABELS, right=False)


# ── MAE table ─────────────────────────────────────────────────────────────────

def mae_table(df, bucket_col, models: dict, title: str):
    """Print a table of MAE/bias/N per bucket for each model."""
    print(f"\n{'=' * 85}")
    print(f"  {title}")
    print(f"{'=' * 85}")
    header = f"  {'Bucket':<22}  {'N':>8}"
    for name in models:
        header += f"  {name:>20s}"
    print(header)
    print(f"  {'-'*22}  {'-'*8}" + "  " + ("  ".join(["-"*20] * len(models))))

    obs = df["observed_rating_delta"].values.astype(float)
    valid_obs = ~np.isnan(obs)

    buckets = df[bucket_col].cat.categories if hasattr(df[bucket_col], "cat") else sorted(df[bucket_col].unique())
    rows = []
    for b in buckets:
        mask = (df[bucket_col] == b) & valid_obs
        n = mask.sum()
        if n < 50:
            continue
        row = {"bucket": b, "n": n}
        for name, pred in models.items():
            r = obs[mask] - pred[mask]
            row[f"{name}_mae"]  = float(np.mean(np.abs(r[~np.isnan(r)])))
            row[f"{name}_bias"] = float(np.mean(r[~np.isnan(r)]))
        rows.append(row)
        names_list = list(models.keys())
        line = f"  {str(b).replace(chr(10),' '):<22}  {n:>8,}"
        for name in names_list:
            mae  = row[f"{name}_mae"]
            bias = row[f"{name}_bias"]
            line += f"  MAE {mae:.3f} / b {bias:+.3f}"
        print(line)

    return rows


# ── plot ──────────────────────────────────────────────────────────────────────

def plot_breakdown(rows_games, rows_mmr, models, fname):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ["steelblue", "tomato", "seagreen"]

    for row_idx, (rows, xlabel, title_suffix) in enumerate([
        (rows_games, "Games this season (before)", "Games played"),
        (rows_mmr,   "Player MMR tier",            "MMR tier"),
    ]):
        buckets = [r["bucket"] for r in rows]
        ns      = [r["n"] for r in rows]
        x       = np.arange(len(buckets))

        # MAE subplot
        ax_mae  = axes[row_idx][0]
        ax_bias = axes[row_idx][1]

        width = 0.8 / len(models)
        for i, name in enumerate(models):
            maes  = [r[f"{name}_mae"]  for r in rows]
            biases= [r[f"{name}_bias"] for r in rows]
            off   = (i - len(models) / 2 + 0.5) * width
            ax_mae.bar( x + off, maes,   width=width*0.9, label=name, color=colors[i], alpha=0.8)
            ax_bias.bar(x + off, biases, width=width*0.9, label=name, color=colors[i], alpha=0.8)

        for ax, ylabel in [(ax_mae, "MAE"), (ax_bias, "Mean bias (obs − pred)")]:
            ax.set_xticks(x)
            ax.set_xticklabels([str(b).replace("\n", " ") for b in buckets],
                               fontsize=8, rotation=20, ha="right")
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(f"{ylabel} by {title_suffix}", fontsize=10)
            ax.legend(fontsize=8)
            if ylabel == "Mean bias (obs − pred)":
                ax.axhline(0, color="gray", lw=0.8, ls="--")

        # Annotate N on MAE bars
        for i, n in enumerate(ns):
            ax_mae.text(x[i], 0.02, f"n={n//1000:.0f}k",
                        ha="center", va="bottom", fontsize=7, color="gray")

    fig.suptitle("Ratings-delta MAE/bias breakdown: P3 formula vs GBT-raw (test set)",
                 fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / fname
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n  → {out}")


# ── joint heatmap: MAE by (MMR tier × games bucket) ──────────────────────────

def plot_heatmap(df, model_preds: dict, fname: str):
    """2-D heatmap of MAE for each (MMR tier × games bucket) cell."""
    obs  = df["observed_rating_delta"].values.astype(float)
    g_b  = assign_games_bucket(df)
    m_b  = assign_mmr_bucket(df)

    n_models = len(model_preds)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
    if n_models == 1:
        axes = [axes]

    for ax, (name, pred) in zip(axes, model_preds.items()):
        mat = np.full((len(MMR_LABELS), len(GAMES_LABELS)), np.nan)
        for ri, ml in enumerate(MMR_LABELS):
            for ci, gl in enumerate(GAMES_LABELS):
                mask = (m_b == ml) & (g_b == gl) & ~np.isnan(obs)
                if mask.sum() < 50:
                    continue
                r = obs[mask] - pred[mask]
                mat[ri, ci] = float(np.nanmean(np.abs(r)))

        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=0.3, vmax=2.5)
        ax.set_xticks(range(len(GAMES_LABELS)))
        ax.set_yticks(range(len(MMR_LABELS)))
        ax.set_xticklabels([l.replace("\n", " ") for l in GAMES_LABELS], fontsize=8)
        ax.set_yticklabels([l.replace("\n", " ") for l in MMR_LABELS], fontsize=8)
        ax.set_xlabel("Games this season (before)", fontsize=9)
        ax.set_ylabel("Player MMR tier", fontsize=9)
        ax.set_title(f"{name} — test MAE heatmap", fontsize=10)
        plt.colorbar(im, ax=ax, label="MAE")

        for ri in range(len(MMR_LABELS)):
            for ci in range(len(GAMES_LABELS)):
                v = mat[ri, ci]
                if not np.isnan(v):
                    ax.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color="black" if v < 1.5 else "white")

    fig.suptitle("MAE heatmap: MMR tier × games played (test set)", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / fname
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    train, valid, test = load_splits()

    # P3 model (saved; load once, predict on all splits)
    print("\nLoading P3 model...")
    p3 = P3Model.load(MODEL_DIR / "p3_parametric.json")
    p3_pred_te = p3.predict(test)
    print(f"  P3 test MAE: {np.nanmean(np.abs(test['observed_rating_delta'] - p3_pred_te)):.4f}")

    # GBT-raw (train + valid → test)
    print("\nTraining GBT-raw...")
    gbt = train_gbt_raw(train, valid)
    X_te = test[[c for c in RAW_FEATURES if c in test.columns]]
    gbt_pred_te = gbt.predict(X_te).astype(float)
    print(f"  GBT-raw test MAE: {np.nanmean(np.abs(test['observed_rating_delta'] - gbt_pred_te)):.4f}")

    models = {
        "P3 formula": p3_pred_te,
        "GBT-raw":    gbt_pred_te,
    }

    # Assign buckets
    test = test.copy()
    test["games_bucket"] = assign_games_bucket(test)
    test["mmr_bucket"]   = assign_mmr_bucket(test)

    # Print tables
    rows_games = mae_table(test, "games_bucket", models,
                           "MAE by games_this_season_before (test set)")
    rows_mmr   = mae_table(test, "mmr_bucket",   models,
                           "MAE by player_mmr_before tier (test set)")

    # Print GBT improvement over P3
    print("\n  GBT improvement over P3 (ΔMAE = P3_MAE − GBT_MAE):")
    print(f"  {'Bucket':<22}  {'ΔMAE (games)':>14}")
    for r in rows_games:
        d = r["P3 formula_mae"] - r["GBT-raw_mae"]
        bar = "█" * max(0, int(d * 20))
        print(f"  {str(r['bucket']).replace(chr(10),' '):<22}  {d:>+.4f}  {bar}")

    print(f"\n  {'Bucket':<22}  {'ΔMAE (MMR)':>14}")
    for r in rows_mmr:
        d = r["P3 formula_mae"] - r["GBT-raw_mae"]
        bar = "█" * max(0, int(d * 20))
        print(f"  {str(r['bucket']).replace(chr(10),' '):<22}  {d:>+.4f}  {bar}")

    # Plots
    print("\nGenerating plots...")
    plot_breakdown(rows_games, rows_mmr, models,
                   "delta_mae_breakdown_bar.png")
    plot_heatmap(test, models, "delta_mae_heatmap.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
