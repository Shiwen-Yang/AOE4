"""
Skill-stratified analysis of civ familiarity predictive value.

Hypothesis: civ familiarity contributes more at lower MMR buckets.
This module tests that hypothesis without assuming it to be true.

New SQL table: player_civ_extra
  - days_since_last_played_civ  (LAG by profile_id, civilization)
  - civ_games_this_patch        (cumulative within profile_id, civilization, patch)

Python-derived:
  - civ_pick_share_{a,b}        civ_games / games_lifetime
  - is_main_civ_{a,b}           civ_pick_share >= 0.20
  - avg_mmr                     (skill_a + skill_b) / 2

Controls (included in all familiarity models):
  - games_lifetime, overall_wr  (experience + baseline skill proxy)
  - days_since                  (recent activity / warmth)
  - civ_pick_share              (normalises civ_games for experience)
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PRIOR_STRENGTH, GLOBAL_WR_PRIOR
from .db import table_exists

BASE_DIR = Path(__file__).resolve().parents[2]
REPORT_PATH = BASE_DIR / "reports" / "generated" / "civ_familiarity_report.md"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

# ── SQL: per-player per-game civ-extra features ────────────────────────────────

_CIV_EXTRA_SQL = """
CREATE OR REPLACE TABLE player_civ_extra AS
SELECT
    p.game_id,
    p.profile_id,
    p.civilization,
    DATEDIFF('day',
        LAG(g.started_at) OVER (
            PARTITION BY p.profile_id, p.civilization
            ORDER BY g.started_at, p.game_id
        ),
        g.started_at
    )                       AS days_since_last_played_civ,
    COALESCE(SUM(1) OVER (
        PARTITION BY p.profile_id, p.civilization, g.patch
        ORDER BY g.started_at, p.game_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0)                   AS civ_games_this_patch
FROM participants p
JOIN games g ON p.game_id = g.game_id
WHERE g.kind IN ('rm_1v1', 'rm_solo')
  AND p.result IS NOT NULL
"""


def build_player_civ_extra(conn) -> None:
    import time
    t0 = time.time()
    conn.execute(_CIV_EXTRA_SQL)
    n = conn.execute("SELECT count(*) FROM player_civ_extra").fetchone()[0]
    print(f"  player_civ_extra: {n:,} rows in {time.time()-t0:.1f}s")


def _fetch_civ_extra(conn) -> pd.DataFrame:
    query = """
    SELECT
        tf.game_id,
        cea.days_since_last_played_civ  AS days_since_civ_a,
        cea.civ_games_this_patch        AS civ_patch_games_a,
        ceb.days_since_last_played_civ  AS days_since_civ_b,
        ceb.civ_games_this_patch        AS civ_patch_games_b
    FROM training_features tf
    LEFT JOIN player_civ_extra cea
        ON tf.game_id = cea.game_id AND tf.profile_id_a = cea.profile_id
    LEFT JOIN player_civ_extra ceb
        ON tf.game_id = ceb.game_id AND tf.profile_id_b = ceb.profile_id
    """
    return conn.execute(query).df()


# ── Python-level derived features ─────────────────────────────────────────────

def _smooth(wins, games):
    p, g = PRIOR_STRENGTH, GLOBAL_WR_PRIOR
    return (wins + p * g) / (games + p)


CIV_FAM_FEATURES = [
    "civ_games_a", "civ_wins_a", "civ_wr_a",
    "civ_games_b", "civ_wins_b", "civ_wr_b",
    "civ_pick_share_a", "civ_pick_share_b",
    "is_main_civ_a", "is_main_civ_b",
    "days_since_civ_a", "days_since_civ_b",
    "civ_patch_games_a", "civ_patch_games_b",
]

CONTROL_FEATURES = [
    "games_lifetime_a", "wins_lifetime_a", "overall_wr_a",
    "games_lifetime_b", "wins_lifetime_b", "overall_wr_b",
    "days_since_a", "days_since_b",
    "games_diff", "wr_diff",
    "is_new_player_a", "is_new_player_b",
    "missing_skill_a", "missing_skill_b",
    "civ_pick_share_a", "civ_pick_share_b",
]

SKILL_ONLY_FEATURES = [
    "mmr_a", "mmr_b", "mmr_diff",
    "rating_a", "rating_b", "rating_diff",
    "skill_a", "skill_b", "skill_diff",
    "missing_mmr_a", "missing_mmr_b",
    "missing_rating_a", "missing_rating_b",
    "missing_skill_a", "missing_skill_b",
]

LIFETIME_FEATURES = [
    "games_lifetime_a", "wins_lifetime_a",
    "games_lifetime_b", "wins_lifetime_b",
    "overall_wr_a", "overall_wr_b",
    "games_diff", "wr_diff",
    "days_since_a", "days_since_b",
    "is_new_player_a", "is_new_player_b",
]

# Civ familiarity features PLUS controls (prevents civ_games acting as experience proxy)
CIV_WITH_CONTROLS = list(dict.fromkeys(CIV_FAM_FEATURES + CONTROL_FEATURES))


def _add_civ_fam_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for side in ("a", "b"):
        glife = df[f"games_lifetime_{side}"].fillna(0).clip(lower=1)
        gciv  = df[f"civ_games_{side}"].fillna(0)
        df[f"civ_pick_share_{side}"] = gciv / glife
        df[f"is_main_civ_{side}"]    = (df[f"civ_pick_share_{side}"] >= 0.20).astype(int)
    df["avg_mmr"] = (df["skill_a"].fillna(0) + df["skill_b"].fillna(0)) / 2.0
    return df


# ── Skill bucket definitions ───────────────────────────────────────────────────

BUCKET_LABELS = ["< 900", "900–1100", "1100–1300", "1300–1500", "≥ 1500"]
BUCKET_BREAKS = [0, 900, 1100, 1300, 1500, 9999]


def _assign_buckets(df: pd.DataFrame) -> pd.Series:
    avg = df["avg_mmr"]
    labels = pd.cut(
        avg,
        bins=BUCKET_BREAKS,
        labels=BUCKET_LABELS,
        right=False,
    )
    return labels


# ── Ablation model training ────────────────────────────────────────────────────

def _train_step(train_df, valid_df, feature_list, cat_cols):
    """Train a LightGBM for one ablation step."""
    import lightgbm as lgb

    # Deduplicate while preserving order (CONTROL_FEATURES overlaps LIFETIME_FEATURES)
    seen = set()
    available = []
    for c in feature_list:
        if c in train_df.columns and c not in seen:
            seen.add(c)
            available.append(c)
    cats = [c for c in cat_cols if c in available]

    params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 63,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "learning_rate": 0.05,
        "min_child_samples": 50,
        "verbose": -1,
        "random_state": 42,
        "feature_pre_filter": False,
    }

    def _ds(df, ref=None):
        X = df[available].copy()
        for c in cats:
            X[c] = X[c].astype("category")
        return lgb.Dataset(X, label=df["target"].values,
                           categorical_feature=cats,
                           reference=ref, free_raw_data=False)

    ds_tr = _ds(train_df)
    ds_va = _ds(valid_df, ref=ds_tr)
    model = lgb.train(
        params, ds_tr, num_boost_round=600,
        valid_sets=[ds_va],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    return model, available, cats


def _metrics(y, p):
    from .evaluate import evaluate
    return evaluate(y, p)


# ── SHAP by bucket ─────────────────────────────────────────────────────────────

def _shap_civ_features(model, df, available, cats, n_sample=2000):
    import shap
    sample = df.sample(min(n_sample, len(df)), random_state=42)
    X = sample[available].copy()
    for c in cats:
        X[c] = X[c].astype("category")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ex = shap.TreeExplainer(model)
        sv = ex.shap_values(X)
    # LightGBM binary returns 2-D array; some shap versions return [neg, pos] list
    if isinstance(sv, list):
        sv = sv[1]
    feat_idx = {f: i for i, f in enumerate(available)}
    out = {}
    for feat in CIV_FAM_FEATURES:
        if feat in feat_idx:
            out[feat] = float(np.abs(sv[:, feat_idx[feat]]).mean())
    return out


# ── Per-bucket ablation ────────────────────────────────────────────────────────

ABLATION_STEPS = [
    ("Skill only",              SKILL_ONLY_FEATURES),
    ("+ lifetime history",      SKILL_ONLY_FEATURES + LIFETIME_FEATURES),
    ("+ civ familiarity",       SKILL_ONLY_FEATURES + LIFETIME_FEATURES + CIV_WITH_CONTROLS),
    ("Full model",              None),   # None = use all available features
]

CAT_COLS = ["civ_a", "civ_b", "map", "patch", "season"]


def _run_bucket_ablation(bucket_label, bucket_df, full_feature_cols, valid_size=0.2):
    """
    Run 4-step ablation on a single skill bucket.
    Temporal order within the bucket is preserved; last valid_size fraction = validation.
    Returns (rows_list, shap_dict).
    """
    n = len(bucket_df)
    if n < 500:
        return [], {}

    bucket_sorted = bucket_df.sort_values("started_at").reset_index(drop=True)
    split = int(n * (1 - valid_size))
    tr = bucket_sorted.iloc[:split]
    va = bucket_sorted.iloc[split:]
    y_va = va["target"].values

    rows = []
    shap_out = {}
    prev_auc = None

    for step_label, feat_list in ABLATION_STEPS:
        effective_feats = feat_list if feat_list is not None else full_feature_cols
        model, used, cats = _train_step(tr, va, effective_feats, CAT_COLS)
        X_va = va[used].copy()
        for c in cats:
            X_va[c] = X_va[c].astype("category")
        preds = model.predict(X_va)
        m = _metrics(y_va, preds)
        delta = (m["auc"] - prev_auc) if prev_auc is not None else None
        rows.append({
            "Skill bucket": bucket_label,
            "Step": step_label,
            "n_valid": len(va),
            "AUC": round(m["auc"], 4),
            "ΔAUC": ("—" if delta is None else f"{delta:+.4f}"),
            "LogLoss": round(m["log_loss"], 4),
            "Brier": round(m["brier"], 4),
        })
        prev_auc = m["auc"]

        # SHAP only for the civ-familiarity step
        if step_label == "+ civ familiarity":
            shap_out = _shap_civ_features(model, va, used, cats)

    return rows, shap_out


# ── Main entrypoint ───────────────────────────────────────────────────────────

def run_civ_familiarity_analysis(conn, df: pd.DataFrame) -> None:
    """
    Full analysis pipeline. conn must be writable (to build player_civ_extra).
    df is the full training features DataFrame with derived features already applied.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Build / refresh civ extra SQL table
    print("Building player_civ_extra table...")
    build_player_civ_extra(conn)

    # 2. Fetch and merge civ extra features
    print("Fetching civ extra features...")
    civ_extra = _fetch_civ_extra(conn)
    df = df.merge(civ_extra, on="game_id", how="left")

    # 3. Python-derived features
    df = _add_civ_fam_derived(df)

    # 4. Temporal split (same 70/15/15 as main model)
    from .model import _temporal_split
    train_df, valid_df, test_df = _temporal_split(df)
    all_df = pd.concat([train_df, valid_df], ignore_index=True)  # train+valid for per-bucket
    del df, train_df, valid_df  # free memory before per-bucket models

    print(f"Dataset: {len(all_df):,} train+valid rows, {len(test_df):,} test rows")

    # 5. Load full model feature list for "Full model" ablation step
    from .model import load_model
    full_model, meta = load_model()
    full_feature_cols = meta["feature_cols"]
    del full_model  # free memory — only need the feature list
    # Add civ extra features not in the saved model
    extra_for_full = [c for c in CIV_FAM_FEATURES if c not in full_feature_cols]
    full_feature_cols = full_feature_cols + extra_for_full

    # 6. Assign skill buckets
    all_df  = all_df.copy();  all_df["skill_bucket"]  = _assign_buckets(all_df)
    test_df = test_df.copy(); test_df["skill_bucket"] = _assign_buckets(test_df)

    bucket_counts = all_df["skill_bucket"].value_counts().sort_index()
    print("Skill bucket distribution (train+valid):")
    for lbl in BUCKET_LABELS:
        print(f"  {lbl:>12}: {bucket_counts.get(lbl, 0):>8,}")

    # 7. Per-bucket ablation
    print(f"\nRunning ablation across {len(BUCKET_LABELS)} skill buckets × {len(ABLATION_STEPS)} steps...")
    all_rows = []
    all_shap: dict[str, dict] = {}  # bucket → {feature: mean_abs_shap}

    for lbl in BUCKET_LABELS:
        mask = all_df["skill_bucket"] == lbl
        bucket_df = all_df[mask]
        print(f"  {lbl}: {mask.sum():,} rows", flush=True)
        rows, shap = _run_bucket_ablation(lbl, bucket_df, full_feature_cols)
        all_rows.extend(rows)
        if shap:
            all_shap[lbl] = shap

    results_df = pd.DataFrame(all_rows)

    # 8. Build civ-lift summary (ΔAUC from adding civ familiarity)
    civ_lift = (
        results_df[results_df["Step"] == "+ civ familiarity"]
        [["Skill bucket", "n_valid", "AUC", "ΔAUC", "Brier"]]
        .reset_index(drop=True)
    )

    # 9. SHAP table: mean |SHAP| per civ feature × bucket
    shap_rows = []
    for lbl, feat_shap in all_shap.items():
        for feat, val in sorted(feat_shap.items(), key=lambda x: -x[1]):
            shap_rows.append({"Skill bucket": lbl, "Feature": feat, "Mean |SHAP|": round(val, 5)})
    shap_df = pd.DataFrame(shap_rows)

    # Pivot: features as rows, buckets as columns
    if not shap_df.empty:
        shap_pivot = shap_df.pivot(index="Feature", columns="Skill bucket", values="Mean |SHAP|").fillna(0)
        shap_pivot = shap_pivot.reindex(columns=[l for l in BUCKET_LABELS if l in shap_pivot.columns])
        shap_pivot["Total"] = shap_pivot.sum(axis=1)
        shap_pivot = shap_pivot.sort_values("Total", ascending=False).drop(columns="Total")
        shap_pivot = shap_pivot.round(5).reset_index()
    else:
        shap_pivot = pd.DataFrame()

    # 10. Figures
    FIGURES_DIR.mkdir(exist_ok=True)

    # Figure A: AUC by step × bucket (line chart)
    fig, ax = plt.subplots(figsize=(10, 5))
    step_labels = [s for s, _ in ABLATION_STEPS]
    for lbl in BUCKET_LABELS:
        sub = results_df[results_df["Skill bucket"] == lbl]
        if sub.empty:
            continue
        aucs = [sub[sub["Step"] == s]["AUC"].values[0] if not sub[sub["Step"] == s].empty else np.nan
                for s in step_labels]
        ax.plot(step_labels, aucs, marker="o", label=lbl)
    ax.set_xticks(range(len(step_labels)))
    ax.set_xticklabels(step_labels, rotation=15, ha="right")
    ax.set_ylabel("Validation AUC")
    ax.set_title("Civ familiarity ablation by skill bucket")
    ax.legend(title="avg MMR bucket")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "civ_auc_by_bucket.png", dpi=120)
    plt.close(fig)

    # Figure B: Civ-lift (ΔAUC) by bucket bar chart
    lift_vals = []
    for lbl in BUCKET_LABELS:
        sub = civ_lift[civ_lift["Skill bucket"] == lbl]
        if sub.empty:
            lift_vals.append(0.0)
        else:
            try:
                v = float(sub["ΔAUC"].values[0])
            except (ValueError, TypeError):
                v = 0.0
            lift_vals.append(v)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    colors = ["#4C72B0" if v >= 0 else "#C44E52" for v in lift_vals]
    ax2.bar(BUCKET_LABELS, lift_vals, color=colors)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("ΔAUC (adding civ familiarity)")
    ax2.set_title("Marginal civ familiarity lift by skill bucket")
    ax2.set_xlabel("avg MMR bucket")
    plt.tight_layout()
    fig2.savefig(FIGURES_DIR / "civ_lift_by_bucket.png", dpi=120)
    plt.close(fig2)

    # 11. Generate report
    _write_report(results_df, civ_lift, shap_pivot, bucket_counts, shap_df)
    print(f"\nReport written to {REPORT_PATH}")


def _md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep    = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows   = "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    )
    return "\n".join([header, sep, rows])


def _write_report(results_df, civ_lift, shap_pivot, bucket_counts, shap_df):
    # Infer hypothesis direction from lift values
    numeric_lifts = []
    for lbl in BUCKET_LABELS:
        sub = civ_lift[civ_lift["Skill bucket"] == lbl]
        if not sub.empty:
            try:
                numeric_lifts.append((lbl, float(sub["ΔAUC"].values[0])))
            except (ValueError, TypeError):
                pass

    if len(numeric_lifts) >= 2:
        low_lift  = next((v for l, v in numeric_lifts if l == BUCKET_LABELS[0]),  None)
        high_lift = next((v for l, v in numeric_lifts if l == BUCKET_LABELS[-1]), None)
        if low_lift is not None and high_lift is not None:
            hypothesis_verdict = (
                "**Supported** — civ familiarity ΔAUC is higher in lower MMR buckets."
                if low_lift > high_lift
                else "**Not supported** — civ familiarity ΔAUC does not decrease monotonically with skill."
            )
        else:
            hypothesis_verdict = "**Inconclusive** — insufficient data in extreme buckets."
    else:
        hypothesis_verdict = "**Inconclusive** — insufficient bucket data."

    lines = [
        "# Civ Familiarity Analysis — Skill-Stratified Report",
        "",
        "**Hypothesis**: civ familiarity features contribute more predictive lift at lower MMR,",
        "and less lift at higher MMR (where players tend to be more settled in their civ choices).",
        "",
        "## 1. Dataset and Skill Buckets",
        "",
        "Avg MMR = (skill_a + skill_b) / 2 where skill = mmr if available, else rating.",
        "Temporal split: first 70% of each bucket = train, last 20% within bucket = validation.",
        "",
        "| Skill bucket | Train+valid rows |",
        "| --- | --- |",
    ]
    for lbl in BUCKET_LABELS:
        lines.append(f"| {lbl} | {bucket_counts.get(lbl, 0):,} |")

    lines += [
        "",
        "## 2. New Civ Familiarity Features",
        "",
        "| Feature | Source | Notes |",
        "| --- | --- | --- |",
        "| `civ_games_a/b` | `player_stats` (existing) | Cumulative lifetime games with this civ |",
        "| `civ_wr_a/b` | Derived (existing) | Smoothed win rate with this civ |",
        "| `civ_pick_share_a/b` | Derived (new) | `civ_games / games_lifetime` — civ concentration |",
        "| `is_main_civ_a/b` | Derived (new) | `civ_pick_share ≥ 0.20` |",
        "| `days_since_civ_a/b` | SQL LAG (new) | Days since last game with THIS civ |",
        "| `civ_patch_games_a/b` | SQL window (new) | Games with this civ in the current patch |",
        "",
        "**Controls included in all civ-familiarity models** (to prevent civ_games acting as an",
        "experience proxy): `games_lifetime`, `overall_wr`, `days_since`, `wr_diff`, `is_new_player`,",
        "`civ_pick_share` (normalises civ_games by total experience).",
        "",
        "## 3. Ablation Results by Skill Bucket",
        "",
    ]

    for lbl in BUCKET_LABELS:
        sub = results_df[results_df["Skill bucket"] == lbl]
        if sub.empty:
            lines += [f"### {lbl}", "", "Insufficient data (< 500 rows).", ""]
            continue
        show = sub[["Step", "n_valid", "AUC", "ΔAUC", "LogLoss", "Brier"]].reset_index(drop=True)
        lines += [f"### {lbl}", "", _md_table(show), ""]

    lines += [
        "## 4. AUC Progression Chart",
        "",
        "![AUC by step and bucket](figures/civ_auc_by_bucket.png)",
        "",
        "## 5. Civ Familiarity Lift Summary",
        "",
        "ΔAUC from adding civ familiarity features (step 3 vs step 2) per skill bucket.",
        "",
        _md_table(civ_lift),
        "",
        "![Civ lift by bucket](figures/civ_lift_by_bucket.png)",
        "",
        "## 6. SHAP — Mean |SHAP| for Civ Familiarity Features",
        "",
        "Computed on the validation set of the `+ civ familiarity` model per bucket.",
        "Higher = stronger absolute contribution to predictions in that bucket.",
        "",
    ]

    if not shap_pivot.empty:
        lines += [_md_table(shap_pivot), ""]
    else:
        lines += ["*SHAP not available (insufficient data).*", ""]

    lines += [
        "## 7. Hypothesis Verdict",
        "",
        hypothesis_verdict,
        "",
    ]

    if len(numeric_lifts) >= 2:
        lines.append("Lift (ΔAUC) by bucket:")
        for lbl, val in numeric_lifts:
            lines.append(f"- **{lbl}**: ΔAUC = {val:+.4f}")

    lines += [
        "",
        "## 8. Interpretation Notes",
        "",
        "- Civ familiarity features are controlled against `games_lifetime`, `overall_wr`, `civ_pick_share`,",
        "  and `days_since` to prevent `civ_games` from acting as a disguised experience proxy.",
        "- `days_since_last_played_civ` captures civ-specific warmth (rust), independent of overall activity.",
        "- `civ_patch_games_this_patch` measures current-patch familiarity, which matters when civs",
        "  change between patches and muscle memory needs to be rebuilt.",
        "- Low MMR players have higher variance outcomes; even a small ΔAUC may be practically meaningful.",
        "- `is_main_civ` (pick_share ≥ 20%) is a coarse threshold; the raw `civ_pick_share` carries more",
        "  gradient information for the model.",
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
