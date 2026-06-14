"""Pair-sum analysis: does P3 correctly account for non-zero-sum behavior?

The game system creates net positive rating in placement / missing-MMR games.
P3 handles this through asymmetric K_win/K_loss and the piecewise b(g) intercept.
This script verifies whether those mechanisms are correctly calibrated at the
pair level — i.e., predicted pair sum ≈ observed pair sum by experience bucket.

Usage (from repo root):
    PYTHONPATH=src python reports/scripts/analyze_pair_sum.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ratings_delta.parametric import P3Model, _intercept_eval, _pieces_to_list

DATA_DIR  = REPO / "reports" / "generated"
FIG_DIR   = REPO / "reports" / "figures"
MODEL_DIR = REPO / "models"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── load ──────────────────────────────────────────────────────────────────────

def _load(path):
    df = pd.read_csv(path, low_memory=False)
    for col in [
        "result", "observed_rating_delta",
        "player_mmr_before", "opponent_mmr_before",
        "player_rating_before", "opponent_rating_before",
        "games_this_season_before", "opponent_games_this_season_before",
        "visible_rating_gap", "hidden_mmr_gap",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── pair-sum helpers ──────────────────────────────────────────────────────────

def build_pairs(df: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Join participant rows by game_id, return one row per complete pair."""
    df = df.copy()
    df["pred"] = pred
    df["resid"] = df["observed_rating_delta"] - df["pred"]

    agg = (
        df.dropna(subset=["observed_rating_delta", "game_id"])
        .groupby("game_id")
        .agg(
            obs_sum    =("observed_rating_delta", "sum"),
            pred_sum   =("pred",                  "sum"),
            resid_sum  =("resid",                 "sum"),
            n_parts    =("profile_id",             "count"),
            min_games  =("games_this_season_before", "min"),
            max_games  =("games_this_season_before", "max"),
            miss_mmr_any=("player_mmr_before",    lambda x: x.isna().any()),
        )
        .reset_index()
    )
    pairs = agg[agg["n_parts"] == 2].copy()
    pairs["pair_resid"] = pairs["obs_sum"] - pairs["pred_sum"]
    return pairs


def bucket_label(min_g: float) -> str:
    if min_g < 10:
        return "placement (< 10)"
    if min_g < 50:
        return "settling (10–49)"
    return "established (≥ 50)"


# ── section 1: pair-sum calibration table ────────────────────────────────────

def calibration_table(pairs: pd.DataFrame, split_name: str):
    print(f"\n=== Pair-sum calibration [{split_name}] ===")
    print(f"  {'Bucket':<24}  {'N':>9}  {'Obs mean':>10}  {'Pred mean':>10}  "
          f"{'Resid mean':>11}  {'|resid|≤1':>10}")

    buckets = [
        ("placement (min_g < 10)",  pairs["min_games"] < 10),
        ("settling  (10–49)",       (pairs["min_games"] >= 10) & (pairs["min_games"] < 50)),
        ("established (≥ 50)",      pairs["min_games"] >= 50),
        ("─" * 24,                  None),
        ("any missing MMR",         pairs["miss_mmr_any"]),
        ("both have MMR",           ~pairs["miss_mmr_any"]),
        ("─" * 24,                  None),
        ("ALL",                     pd.Series(True, index=pairs.index)),
    ]
    for label, mask in buckets:
        if mask is None:
            print(f"  {label}")
            continue
        sub = pairs[mask]
        if not len(sub):
            continue
        pct = (sub["pair_resid"].abs() <= 1).mean() * 100
        print(f"  {label:<24}  {len(sub):>9,}  "
              f"{sub['obs_sum'].mean():>+10.3f}  "
              f"{sub['pred_sum'].mean():>+10.3f}  "
              f"{sub['pair_resid'].mean():>+11.3f}  "
              f"{pct:>9.1f}%")


# ── section 2: obs vs pred pair sum by game count (binned) ───────────────────

def plot_pair_sum_by_game_count(pairs: pd.DataFrame, fname: str):
    """Plot observed vs predicted pair sum as a function of min(games) per game."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (lo, hi, title) in zip(axes, [
        (0,  50, "Placement / settling range (min_g 0–49)"),
        (50, 200, "Established range (min_g 50–200)"),
    ]):
        sub = pairs[(pairs["min_games"] >= lo) & (pairs["min_games"] < hi)].copy()
        if len(sub) < 100:
            ax.set_visible(False)
            continue

        bins = np.arange(lo, hi + 1, 1)
        bin_idx = np.digitize(sub["min_games"].values, bins) - 1
        bx, obs_m, pred_m, resid_m = [], [], [], []
        for i in range(len(bins) - 1):
            sel = sub.iloc[bin_idx == i]
            if len(sel) < 30:
                continue
            bx.append(bins[i])
            obs_m.append(sel["obs_sum"].mean())
            pred_m.append(sel["pred_sum"].mean())
            resid_m.append(sel["pair_resid"].mean())

        bx = np.array(bx)
        ax.plot(bx, obs_m,   lw=2,   color="steelblue", label="Observed pair sum")
        ax.plot(bx, pred_m,  lw=2,   color="tomato",    label="P3 predicted pair sum")
        ax.plot(bx, resid_m, lw=1.5, color="green", ls="--", label="Residual (obs − pred)")
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.set_xlabel("min(games_this_season) in pair")
        ax.set_ylabel("Mean pair rating sum")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("Pair-sum calibration: P3 observed vs predicted", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / fname
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  → {out}")


# ── section 3: b(g) curve vs empirical mean delta ────────────────────────────

def plot_bg_vs_empirical(df: pd.DataFrame, p3: P3Model, fname: str):
    """Overlay the fitted b(g) curve against the per-game-count mean K-residual."""
    # K/D residual per participant: obs - K*(y-p) = obs - (pred - b(g) - beta0 - missing_corr)
    # Simpler: just look at mean (obs - pred_no_bg) per game count, where pred_no_bg
    # = the P2 K/D-only prediction (beta0 + K*(y-p)), excluding b(g) and gamma/missing terms.

    from ratings_delta.parametric import _ra_rb, _obs, elo_expected

    ra, rb  = _ra_rb(df)
    obs     = _obs(df)
    res     = df["result"].values.astype(float)
    games   = df["games_this_season_before"].to_numpy(dtype=float, na_value=np.nan)
    p2      = p3.p2
    p       = elo_expected(ra, rb, p2.D)

    t0, t1  = p2._THRESHOLDS
    bmasks  = [games < t0, (games >= t0) & (games < t1), games >= t1]

    kd = np.full(len(df), np.nan)
    for bm, kw, kl in zip(bmasks, p2.K_win, p2.K_loss):
        kd[bm] = np.where(
            res[bm] == 1,
            p2.beta0 + kw * (1 - p[bm]),
            p2.beta0 + kl * (-p[bm]),
        )
    kd_resid = obs - kd  # = b(g) + gamma_correction + alpha_missing + noise

    # Filter to MMR-present rows so the empirical mean is directly comparable to
    # b(g), which was fitted on MMR-present rows only.  Missing-MMR rows carry an
    # extra −8.6 offset (alpha_missing_p) that would otherwise create a spurious
    # gap in the plot at low n where missing-MMR fraction is high (~50% in test).
    mmr_p = df["player_mmr_before"].to_numpy(dtype=float, na_value=np.nan) if "player_mmr_before" in df.columns else np.full(len(df), np.nan)
    mmr_present = ~np.isnan(mmr_p)

    g_int = np.where(~np.isnan(games), games.astype(int), -1)
    tbl = (
        pd.DataFrame({"g": g_int, "r": kd_resid, "keep": mmr_present})
        .query("keep and g >= 0 and g <= 150")
        .drop(columns="keep")
        .dropna()
        .groupby("g")["r"]
        .agg(["mean", "count"])
        .reset_index()
    )
    tbl = tbl[tbl["count"] >= 50]

    # Fitted b(g) curve
    g_range = np.arange(0, 151, 1, dtype=float)
    b_curve = _intercept_eval(g_range, p2.pieces)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (lo, hi, title) in zip(axes, [
        (0,  50, "Placement / settling (g 0–49)"),
        (50, 150, "Established (g 50–150)"),
    ]):
        mask_tbl   = (tbl["g"] >= lo) & (tbl["g"] < hi)
        mask_curve = (g_range >= lo) & (g_range < hi)

        ax.scatter(tbl["g"][mask_tbl], tbl["mean"][mask_tbl],
                   s=tbl["count"][mask_tbl] / tbl["count"].max() * 80 + 5,
                   alpha=0.6, color="steelblue", label="Empirical mean K-residual")
        ax.plot(g_range[mask_curve], b_curve[mask_curve],
                lw=2, color="tomato", label="Fitted b(g) curve")
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.set_xlabel("games_this_season_before")
        ax.set_ylabel("Mean K-residual, MMR-present rows only")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("b(g) curve vs empirical per-game-count residual (test set)", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / fname
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  → {out}")


# ── section 4: pair-sum residual decomposition ───────────────────────────────

def pair_resid_decomposition(pairs: pd.DataFrame):
    """Break the pair-sum residual into K-contribution vs b(g)-contribution."""
    print("\n=== Pair-sum residual decomposition ===")
    print("  Non-zero-sum arises from two model components:")
    print("  (A) Asymmetric K: K_win ≠ K_loss creates non-zero K-term pair sum")
    print("  (B) Piecewise b(g): b(g_p) + b(g_opp) adds per-game intercept pair sum")

    # For each bucket, explain what fraction of observed pair sum is accounted for
    buckets = [
        ("placement  (min_g <  10)", pairs["min_games"] < 10),
        ("settling   (min_g 10–49)", (pairs["min_games"] >= 10) & (pairs["min_games"] < 50)),
        ("established (min_g ≥ 50)", pairs["min_games"] >= 50),
    ]
    print(f"\n  {'Bucket':<26}  {'Obs mean':>10}  {'Pred mean':>10}  "
          f"{'Frac explained':>15}")
    for label, mask in buckets:
        sub = pairs[mask]
        if not len(sub):
            continue
        obs_m  = sub["obs_sum"].mean()
        pred_m = sub["pred_sum"].mean()
        frac   = pred_m / obs_m * 100 if abs(obs_m) > 0.1 else float("nan")
        print(f"  {label:<26}  {obs_m:>+10.3f}  {pred_m:>+10.3f}  {frac:>14.1f}%")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading P3 model and test data...")
    p3   = P3Model.load(MODEL_DIR / "p3_parametric.json")
    test = _load(DATA_DIR / "ratings_delta_test.csv")
    train = _load(DATA_DIR / "ratings_delta_train.csv")

    print(f"  Test rows: {len(test):,}  |  Train rows: {len(train):,}")

    # Predictions
    pred_te = p3.predict(test)
    pred_tr = p3.predict(train)

    # Build pair tables
    pairs_te = build_pairs(test,  pred_te)
    pairs_tr = build_pairs(train, pred_tr)
    print(f"  Complete test pairs:  {len(pairs_te):,}")
    print(f"  Complete train pairs: {len(pairs_tr):,}")

    # Section 1: calibration tables
    calibration_table(pairs_tr, "train")
    calibration_table(pairs_te, "test")

    # Section 2: obs vs pred pair-sum curve
    print("\nPlotting pair-sum calibration curves...")
    plot_pair_sum_by_game_count(pairs_te, "pair_sum_calibration.png")

    # Section 3: b(g) curve vs empirical residual
    print("Plotting b(g) curve vs empirical K-residual...")
    plot_bg_vs_empirical(test, p3, "bg_curve_vs_empirical.png")

    # Section 4: decomposition
    pair_resid_decomposition(pairs_te)

    print("\nDone.")


if __name__ == "__main__":
    main()
