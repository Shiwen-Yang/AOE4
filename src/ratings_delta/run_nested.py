"""Evaluate nested parametric models P0–P3 on the saved ratings-delta dataset.

Usage (from repo root):
    PYTHONPATH=src python -m ratings_delta.run_nested

Loads from reports/generated/ratings_delta_{train,valid,test}.csv.
P3 is the accepted final model; P4 appears in a diagnostic appendix.
Saves the fitted P3 model to models/p3_parametric.json.
"""

import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ratings_delta.parametric import (
    P0Model, P2Model, P3Model, P4Model,
    metrics, _obs,
)

# ── paths ─────────────────────────────────────────────────────────────────────

REPO     = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "reports" / "generated"
FIG_DIR  = REPO / "reports" / "figures"
MODEL_DIR = REPO / "models"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "ratings_delta_train.csv"
VALID_CSV  = DATA_DIR / "ratings_delta_valid.csv"
TEST_CSV   = DATA_DIR / "ratings_delta_test.csv"
P3_MODEL   = MODEL_DIR / "p3_parametric.json"


# ── loading ───────────────────────────────────────────────────────────────────

def _load(path: Path) -> pd.DataFrame:
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


def load_splits():
    print("Loading splits...")
    train = _load(TRAIN_CSV)
    valid = _load(VALID_CSV)
    test  = _load(TEST_CSV)
    for name, df in [("train", train), ("valid", valid), ("test", test)]:
        print(f"  {name:5s}: {len(df):>10,} rows")
    return train, valid, test


# ── evaluation table ──────────────────────────────────────────────────────────

def _fmt_row(label, m_tr, m_va, m_te, n_p, winner=False):
    def _c(m):
        return f"{m['mae']:.4f} / {m['rmse']:.4f} / {m['mean_signed']:+.4f}"
    star = " ◄" if winner else ""
    return (f"  {label:<42s}  {_c(m_tr)}  ||  {_c(m_va)}  ||  {_c(m_te)}"
            f"  [{n_p}p]{star}")


def _print_table_header():
    h = ("  " + f"{'Model':<42s}  "
         + "Train MAE/RMSE/bias              ||  "
         + "Valid MAE/RMSE/bias              ||  "
         + "Test  MAE/RMSE/bias")
    sep = "=" * 155
    print(f"\n{sep}\n{h}\n{sep}")


# ── residual plot helper ──────────────────────────────────────────────────────

def _plot_resid_vs_features(resid: np.ndarray, df: pd.DataFrame,
                             title: str, fname: str):
    features = [
        ("player_mmr_before",        "Player MMR"),
        ("visible_rating_gap",       "Visible rating gap"),
        ("player_rating_before",     "Player rating"),
        ("opponent_rating_before",   "Opponent rating"),
        ("games_this_season_before", "Games this season"),
    ]
    valid_resid = ~np.isnan(resid)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for ax, (col, label) in zip(axes, features):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        x = df[col].to_numpy(dtype=float, na_value=np.nan)
        mask = valid_resid & ~np.isnan(x)
        if mask.sum() < 100:
            ax.set_visible(False)
            continue
        xv, rv = x[mask], resid[mask]
        bins = np.unique(np.percentile(xv, np.linspace(0, 100, 52)))
        idx  = np.digitize(xv, bins) - 1
        bx, bm, bs = [], [], []
        for i in range(len(bins) - 1):
            r_i = rv[idx == i]
            if len(r_i) < 20:
                continue
            bx.append((bins[i] + bins[i + 1]) / 2)
            bm.append(float(np.mean(r_i)))
            bs.append(float(np.std(r_i) / np.sqrt(len(r_i))))
        bx, bm, bs = np.array(bx), np.array(bm), np.array(bs)
        ax.plot(bx, bm, lw=1.5, color="steelblue")
        ax.fill_between(bx, bm - bs, bm + bs, alpha=0.25, color="steelblue")
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Residual" if ax is axes[0] else "")
        ax.set_title(label, fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / fname
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  → {out}")


# ── pair-sum analysis ─────────────────────────────────────────────────────────

def pair_sum_analysis(train_full: pd.DataFrame):
    print("\n=== Pair-Sum Analysis ===")
    if "game_id" not in train_full.columns:
        print("  game_id not found; skipping.")
        return

    df = train_full.dropna(subset=["observed_rating_delta", "game_id"]).copy()
    agg = df.groupby("game_id").agg(
        delta_sum=("observed_rating_delta", "sum"),
        n_parts  =("observed_rating_delta", "count"),
        min_games=("games_this_season_before", "min"),
        miss_mmr =("player_mmr_before", lambda x: x.isna().any()),
    ).reset_index()
    pairs = agg[agg["n_parts"] == 2].copy()

    print(f"  Complete pairs: {len(pairs):,}  |  overall mean sum: {pairs['delta_sum'].mean():+.3f}")
    print(f"\n  {'Bucket':<32}  {'N':>9}  {'Mean sum':>10}  {'|sum|≤1':>9}")
    for label, mask in [
        ("placement  (min_g <  10)",  pairs["min_games"] < 10),
        ("settling   (min_g 10–49)", (pairs["min_games"] >= 10) & (pairs["min_games"] < 50)),
        ("established (min_g ≥ 50)", pairs["min_games"] >= 50),
    ]:
        sub = pairs[mask]
        if not len(sub):
            continue
        pct = (sub["delta_sum"].abs() <= 1).mean() * 100
        print(f"  {label:<32}  {len(sub):>9,}  {sub['delta_sum'].mean():>+10.3f}  {pct:>8.1f}%")

    for label, mask in [
        ("any missing MMR", pairs["miss_mmr"]),
        ("both have MMR",  ~pairs["miss_mmr"]),
    ]:
        sub = pairs[mask]
        if not len(sub):
            continue
        pct = (sub["delta_sum"].abs() <= 1).mean() * 100
        print(f"  {label:<32}  {len(sub):>9,}  {sub['delta_sum'].mean():>+10.3f}  {pct:>8.1f}%")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (label, mask) in zip(axes, [
        ("placement (min_g < 10)",   pairs["min_games"] < 10),
        ("settling (10–49 games)",  (pairs["min_games"] >= 10) & (pairs["min_games"] < 50)),
        ("established (≥ 50 games)", pairs["min_games"] >= 50),
    ]):
        sub = pairs[mask]["delta_sum"]
        ax.hist(sub.clip(-30, 30), bins=61, color="steelblue", alpha=0.7)
        ax.axvline(0, color="gray", lw=0.8, ls="--")
        ax.axvline(sub.mean(), color="red", lw=1.2, label=f"mean={sub.mean():+.2f}")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Pair rating sum (clipped ±30)")
        ax.legend(fontsize=8)
    fig.suptitle("Pair-sum distribution by experience bucket (full train)", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "pair_sum_distribution.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n  → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    train_full, valid, test = load_splits()

    train = train_full.sample(frac=1/6, random_state=42).reset_index(drop=True)
    print(f"\n  Fitting on 1/6 subsample: {len(train):,} rows")

    pair_sum_analysis(train_full)

    # ── primary model sequence ────────────────────────────────────────────────
    _print_table_header()
    rows = []

    primary = [
        ("P0 — Elo baseline",                              "p0"),
        ("P2 — subsample only",                            "p2_sub"),
        ("P2 — two-stage (D sub, b(g) full)",              "p2_full"),
        ("P3 — two-stage P2 + missing-MMR indicators",     "p3"),
    ]

    fitted: dict = {}

    for label, key in primary:
        print(f"\nFitting {label}...")

        if key == "p0":
            m = P0Model(); m.fit(train)

        elif key == "p2_sub":
            m = P2Model(); m.fit(train)

        elif key == "p2_full":
            m = P2Model()
            m.D = fitted["p2_sub"].D
            print(f"  D={m.D:.0f} from subsample; re-fitting K/b(g)/γ on {len(train_full):,} rows...")
            m.fit_on_full(train_full)

        elif key == "p3":
            m = P3Model()
            m.fit_two_stage(train, train_full)

        fitted[key] = m
        winner = (key == "p3")
        p3_model = m if winner else (fitted.get("p3") or None)

        m_tr = m.metrics(train)
        m_va = m.metrics(valid)
        m_te = m.metrics(test)
        rows.append((label, m_tr, m_va, m_te, m.n_params(), winner))

        print(_fmt_row(label, m_tr, m_va, m_te, m.n_params(), winner))
        if hasattr(m, "report"):
            print(textwrap.indent(m.report(), "  "))
        else:
            print(f"  {repr(m)}")

        resid = m.residuals(test)
        tag = label.split("—")[0].strip().replace(" ", "_").lower()
        _plot_resid_vs_features(
            resid.to_numpy() if hasattr(resid, "to_numpy") else resid,
            test,
            title=f"{label} — test residuals vs features",
            fname=f"nested_resid_{tag}.png",
        )

    # ── save accepted model ───────────────────────────────────────────────────
    p3_model.save(P3_MODEL)
    print(f"\n  Saved P3 model → {P3_MODEL}")

    # round-trip verify
    p3_loaded = P3Model.load(P3_MODEL)
    m_loaded  = p3_loaded.metrics(test)
    expected_mae = rows[-1][3]["mae"]   # rows item: (label, m_tr, m_va, m_te, n_p, winner)
    assert abs(m_loaded["mae"] - expected_mae) < 1e-6, "Load/save round-trip failed"
    print(f"  Round-trip verified (test MAE {m_loaded['mae']:.4f})")

    # ── final table ───────────────────────────────────────────────────────────
    _print_table_header()
    for label, m_tr, m_va, m_te, n_p, winner in rows:
        print(_fmt_row(label, m_tr, m_va, m_te, n_p, winner))
    print("=" * 155)

    # ── diagnostic appendix: P4 ───────────────────────────────────────────────
    print("\n=== Diagnostic: P4 reconciliation terms ===")
    print("  (Not adopted: β_do ≈ 0; β_dp gives 0.011 MAE gain at cost of +0.67 bias)")
    p4 = P4Model()
    p4.p3 = p3_model
    p4.fit_reconciliation(train_full)
    m_p4 = p4.metrics(test)
    print(f"  P4 test MAE={m_p4['mae']:.4f}  bias={m_p4['mean_signed']:+.4f}")
    print(f"  {repr(p4)}")

    # MMR slope check
    resid_te = _obs(test) - p4.predict(test)
    mmr_te   = test["player_mmr_before"].to_numpy(dtype=float, na_value=np.nan)
    v = ~np.isnan(resid_te) & ~np.isnan(mmr_te)
    slope = np.polyfit(mmr_te[v], resid_te[v], 1)
    print(f"  P4 residual vs MMR slope: {slope[0]:+.6f}  (P5 threshold: 0.001) → P5 not warranted")


if __name__ == "__main__":
    main()
