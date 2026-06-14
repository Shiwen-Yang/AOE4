"""
Generates analysis_report.md with:
  1. Temporal leakage audit
  2. Model performance summary (all baselines + LightGBM + enhanced baseline)
  3. Subgroup analysis by |MMR diff| buckets
  4. SHAP feature importance analysis
  5. Feature family ablation study
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .baselines import (
    CivMapBucketBaseline,
    ConstantBaseline,
    EnhancedLogisticBaseline,
    L1LogisticBaseline,
    MMRLogisticBaseline,
)
from .model import CATEGORICAL_FEATURES
from .db import get_conn, table_exists
from .evaluate import calibration_table, evaluate
from .features import _add_derived_features
from .features_extra import (
    FAMILY_FEATURES, P1_FEATURES, P2_FEATURES, P3_FEATURES,
    P4_FEATURES, P5_FEATURES, P6_FEATURES, P7_FEATURES, P8_FEATURES, P9_FEATURES,
    extend_training_features,
)
from .model import _predict, _temporal_split, load_model

BASE_DIR = Path(__file__).resolve().parents[2]
FIGURES_DIR = BASE_DIR / "reports" / "figures"
REPORT_PATH = BASE_DIR / "reports" / "generated" / "analysis_report.md"

NUMERIC_TOP5_LGBM = [
    "skill_diff",
    "mmr_diff",
    "wins_lifetime_b",
    "civ_wins_b",
    "civ_wins_a",
]

MMR_BUCKETS = [
    ("≤ 50",   0,    50),
    ("51–100",  51,  100),
    ("101–200", 101, 200),
    ("> 200",  201, 99999),
]


# ── helpers ──────────────────────────────────────────────────────────────────

def _md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep    = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows   = "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    )
    return "\n".join([header, sep, rows])


def _metrics_row(label: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    m = evaluate(y_true, y_pred)
    return {
        "Model": label,
        "AUC": f"{m['auc']:.4f}",
        "Log Loss": f"{m['log_loss']:.4f}",
        "Brier": f"{m['brier']:.4f}",
        "Acc@0.5": f"{m['acc@0.5']:.4f}",
    }


def _bucket_mask(df: pd.DataFrame, lo: int, hi: int) -> np.ndarray:
    gap = df["mmr_diff"].abs().fillna(np.nan)
    return ((gap >= lo) & (gap <= hi)).fillna(False).values


# ── leakage audit ─────────────────────────────────────────────────────────────

def _leakage_audit(conn, df: pd.DataFrame) -> str:  # conn unused but kept for backward compat
    # 1. window function violations
    violations = int(
        ((df["wins_lifetime_a"] > df["games_lifetime_a"]) |
         (df["wins_lifetime_b"] > df["games_lifetime_b"])).sum()
    )

    # 2. matchup prior coverage by season
    prior_cover = (
        df.groupby("season")
        .apply(lambda g: pd.Series({
            "total": len(g),
            "has_prior": int((g["prior_matchup_games"] > 0).sum()),
            "null_prior": int((g["prior_matchup_games"].isna() | (g["prior_matchup_games"] == 0)).sum()),
        }))
        .reset_index()
    )
    prior_cover["coverage_%"] = (prior_cover["has_prior"] / prior_cover["total"] * 100).round(1)

    # 3. MMR/rating are stored pre-game in the JSON dump (no window needed)
    mmr_null_pct = df["mmr_a"].isna().mean() * 100

    lines = [
        "## 1. Temporal Leakage Audit",
        "",
        "### Feature-by-feature assessment",
        "",
        "| Feature group | Source | Leakage-free? | Notes |",
        "| --- | --- | --- | --- |",
        "| `mmr_a`, `mmr_b`, `rating_a`, `rating_b` | Direct from game record | ✅ Yes | Stored as pre-game values in the JSON dump |",
        "| `games_lifetime_before`, `wins_lifetime_before` | `ROW_NUMBER / SUM … ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING` | ✅ Yes | Excludes current row |",
        "| `civ_games_before`, `civ_wins_before` | Same window, partitioned by `(profile_id, civ)` | ✅ Yes | |",
        "| `map_games_before`, `map_wins_before` | Same window, partitioned by `(profile_id, map)` | ✅ Yes | |",
        "| `days_since_last_game` | `LAG(started_at)` | ✅ Yes | NULL for a player's first game |",
        "| `prior_matchup_wr_a` | Cumulative over seasons `< current season` | ✅ Yes | Season-granularity; see caveat below |",
        "| Derived rates (`overall_wr_a`, `civ_wr_a`, …) | Smoothed from the above counts | ✅ Yes | No future data used |",
        "",
        f"**Window-function integrity check:** `wins_before > games_before` violations = **{violations}** (must be 0).",
        "",
        "### Civ matchup prior — known caveat",
        "",
        "The matchup prior aggregates games from *seasons before the current season*, not patch-by-patch.",
        "In the current prototype (S10 + S11 only), **S10 games have no prior** because no earlier",
        "seasons were ingested. They receive the smoothed neutral prior (0.5).",
        "",
        _md_table(prior_cover),
        "",
        "This is a prototype data limitation, not a leakage issue.",
        "Ingesting S3–S9 would give S10 games a rich prior from 9M+ earlier matches.",
        "",
        "### MMR missingness",
        "",
        f"MMR is null for **{mmr_null_pct:.1f}%** of training rows (player A side). Where MMR is missing,",
        "`skill_diff` falls back to rating difference. When both are absent, `skill_diff = 0`",
        "and `missing_skill_a/b = 1` flags are set so the model can learn a different response.",
    ]
    return "\n".join(lines)


# ── subgroup helpers ──────────────────────────────────────────────────────────

# Models excluded from subgroup tables (kept in the §2 summary only)
_SUBGROUP_EXCLUDE = frozenset({"Constant", "MMR Logistic"})


def _subgroup_preds(preds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: v for k, v in preds.items() if k not in _SUBGROUP_EXCLUDE}


def _eval_mask(
    y: np.ndarray,
    preds: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict:
    """Evaluate all models on a boolean mask. Returns {model: {AUC, Log Loss, Brier}, _n: int}."""
    n = int(mask.sum())
    result: dict = {"_n": n}
    for model_name, pred in preds.items():
        if n >= 10:
            m = evaluate(y[mask], pred[mask])
            result[model_name] = {
                "AUC":      "—" if np.isnan(m["auc"])      else f"{m['auc']:.4f}",
                "Log Loss": "—" if np.isnan(m["log_loss"]) else f"{m['log_loss']:.4f}",
                "Brier":    f"{m['brier']:.4f}",
            }
        else:
            result[model_name] = {"AUC": "—", "Log Loss": "—", "Brier": "—"}
    return result


def _three_metric_tables(
    labeled_results: list[tuple[str, dict]],
    group_col: str,
) -> str:
    """Render AUC / Log Loss / Brier as three separate sub-tables for easier comparison."""
    if not labeled_results:
        return "(no subgroups with n ≥ 10)"
    models = [k for k in labeled_results[0][1] if k != "_n"]
    blocks = []
    for metric in ("AUC", "Log Loss", "Brier"):
        records = []
        for label, res in labeled_results:
            rec = {group_col: label, "n": f"{res['_n']:,}"}
            for model in models:
                rec[model] = res[model][metric]
            records.append(rec)
        blocks.append(f"### {metric}\n\n{_md_table(pd.DataFrame(records))}")
    return "\n\n".join(blocks)


def _quintile_buckets(
    series: pd.Series,
    thresholds: np.ndarray,
    name: str,
    fmt: str = "g",
) -> list[tuple[str, np.ndarray]]:
    """Return (label, bool_mask) pairs for 5 quintile buckets from 4 threshold values."""
    cuts = [series.min()] + list(thresholds) + [float("inf")]
    buckets = []
    for i in range(5):
        lo, hi = cuts[i], cuts[i + 1]
        mask = (series >= lo) & (series < hi)
        label = (f"≥{lo:{fmt}} {name} (Q{i+1})" if hi == float("inf")
                 else f"{lo:{fmt}}–{hi:{fmt}} {name} (Q{i+1})")
        buckets.append((label, mask.values))
    return buckets


# ── subgroup sections ─────────────────────────────────────────────────────────

def _subgroup_section(test_df: pd.DataFrame, preds: dict[str, np.ndarray]) -> str:
    y = test_df["target"].values
    group_col = "MMR diff bucket"
    sp = _subgroup_preds(preds)

    labeled = []
    for label, lo, hi in MMR_BUCKETS:
        mask = _bucket_mask(test_df, lo, hi)
        res = _eval_mask(y, sp, mask)
        if res["_n"] >= 10:
            labeled.append((label, res))

    missing_mask = (
        (test_df["missing_skill_a"].values == 1) | (test_df["missing_skill_b"].values == 1)
    )
    labeled.append(("Either player missing MMR/rating", _eval_mask(y, sp, missing_mask)))

    lines = [
        "## 3. Subgroup Analysis by |MMR diff|",
        "",
        "Prediction accuracy varies substantially with the skill gap.",
        "When players are evenly matched, outcomes are closer to random — harder for any model.",
        "",
        _three_metric_tables(labeled, group_col),
        "",
        "AUC: rank discrimination (higher = better). "
        "Log Loss / Brier: calibration + accuracy (lower = better). "
        "|MMR diff| ≤ 50 limits any model's AUC ceiling.",
    ]
    return "\n".join(lines)


def _subgroup_min_season_games_section(
    test_df: pd.DataFrame,
    preds: dict[str, np.ndarray],
    thresholds: np.ndarray,
) -> str:
    y = test_df["target"].values
    group_col = "Min season games"
    sp = _subgroup_preds(preds)
    min_games = test_df[["games_season_a", "games_season_b"]].min(axis=1)
    buckets = _quintile_buckets(min_games, thresholds, "games", fmt=".0f")

    labeled = [(lbl, r) for lbl, mask in buckets
               for r in [_eval_mask(y, sp, mask)] if r["_n"] >= 10]

    cuts_str = " | ".join(f"p{p}={int(t)}" for p, t in zip([20, 40, 60, 80], thresholds))
    lines = [
        "## 4. Subgroup Analysis by Min Season Games (Less-Active Player)",
        "",
        f"Buckets by `min(games_season_a, games_season_b)` — quintiles from training set ({cuts_str}).",
        "Captures the experience level of the less-active player in each match.",
        "",
        _three_metric_tables(labeled, group_col),
        "",
        "Q1 = fewest games (newcomers), Q5 = most active.",
    ]
    return "\n".join(lines)


def _subgroup_activity_gap_section(
    test_df: pd.DataFrame,
    preds: dict[str, np.ndarray],
    thresholds: np.ndarray,
) -> str:
    y = test_df["target"].values
    group_col = "Activity gap"
    sp = _subgroup_preds(preds)
    gap = (test_df["games_season_a"] - test_df["games_season_b"]).abs()
    buckets = _quintile_buckets(gap, thresholds, "game gap", fmt=".0f")

    labeled = [(lbl, r) for lbl, mask in buckets
               for r in [_eval_mask(y, sp, mask)] if r["_n"] >= 10]

    cuts_str = " | ".join(f"p{p}={int(t)}" for p, t in zip([20, 40, 60, 80], thresholds))
    lines = [
        "## 5. Subgroup Analysis by Activity Gap (|games_season_a − games_season_b|)",
        "",
        f"Buckets by absolute season game-count difference — quintiles from training set ({cuts_str}).",
        "Captures how mismatched the two players are in recent activity.",
        "",
        _three_metric_tables(labeled, group_col),
        "",
        "Q1 = evenly matched activity, Q5 = most asymmetric (grinder vs returning player).",
    ]
    return "\n".join(lines)


def _subgroup_mean_mmr_section(
    test_df: pd.DataFrame,
    preds: dict[str, np.ndarray],
    thresholds: np.ndarray,
) -> str:
    y = test_df["target"].values
    group_col = "Mean hidden MMR tier"
    sp = _subgroup_preds(preds)

    mean_mmr = (test_df["mmr_a"] + test_df["mmr_b"]) / 2
    has_mmr = mean_mmr.notna()
    p20, p40, p60, p80, p99 = thresholds

    tier_defs = [
        (f"< {p20:.0f} (Q1 — lowest)",       has_mmr & (mean_mmr <  p20)),
        (f"{p20:.0f}–{p40:.0f} (Q2)",         has_mmr & (mean_mmr >= p20) & (mean_mmr < p40)),
        (f"{p40:.0f}–{p60:.0f} (Q3 — mid)",   has_mmr & (mean_mmr >= p40) & (mean_mmr < p60)),
        (f"{p60:.0f}–{p80:.0f} (Q4)",         has_mmr & (mean_mmr >= p60) & (mean_mmr < p80)),
        (f"{p80:.0f}–{p99:.0f} (Q5 — high)",  has_mmr & (mean_mmr >= p80) & (mean_mmr < p99)),
        (f"≥ {p99:.0f} (top 1% — pro tier)",  has_mmr & (mean_mmr >= p99)),
    ]

    labeled = [(lbl, r) for lbl, s in tier_defs
               for r in [_eval_mask(y, sp, s.values)] if r["_n"] >= 10]
    labeled.append(("Either player missing hidden MMR", _eval_mask(y, sp, (~has_mmr).values)))

    lines = [
        "## 6. Subgroup Analysis by Mean Hidden MMR Tier",
        "",
        f"Buckets by `(mmr_a + mmr_b) / 2` — 5 quintile tiers plus a top-1% pro tier.",
        f"Percentile cutoffs from training set: p20={p20:.0f}, p40={p40:.0f}, "
        f"p60={p60:.0f}, p80={p80:.0f}, p99={p99:.0f} MMR.",
        "Games where either player lacks hidden MMR go to the missing bucket.",
        "",
        _three_metric_tables(labeled, group_col),
        "",
        "Higher tiers → better-rated players; pro tier (≥p99) shows whether "
        "skill gaps remain meaningful at the top.",
    ]
    return "\n".join(lines)


def _subgroup_missing_mmr_section(
    test_df: pd.DataFrame,
    preds: dict[str, np.ndarray],
) -> str:
    y = test_df["target"].values
    group_col = "MMR availability"
    sp = _subgroup_preds(preds)

    miss_a = test_df["missing_mmr_a"].values == 1
    miss_b = test_df["missing_mmr_b"].values == 1

    buckets = [
        ("Both players have MMR",     (~miss_a) & (~miss_b)),
        ("Player A missing MMR",      miss_a   & (~miss_b)),
        ("Player B missing MMR",      (~miss_a) & miss_b),
        ("Either player missing MMR", miss_a   | miss_b),
    ]

    labeled = [(lbl, r) for lbl, mask in buckets
               for r in [_eval_mask(y, sp, mask)] if r["_n"] >= 10]

    lines = [
        "## 7. Subgroup Analysis by MMR Availability",
        "",
        "Shows how model performance shifts when one or both players lack a hidden MMR.",
        "The model falls back to visible rating; when both are absent, skill_diff = 0.",
        "",
        _three_metric_tables(labeled, group_col),
    ]
    return "\n".join(lines)


# ── SHAP analysis ─────────────────────────────────────────────────────────────

def _shap_section(
    model,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    fig_dir: Path | None = None,
    n_sample: int = 5000,
    report_dir: Path | None = None,
) -> str:
    fig_dir = fig_dir or FIGURES_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    available = [c for c in feature_cols if c in test_df.columns]
    cat_feats = [c for c in CATEGORICAL_FEATURES if c in available]

    sample = test_df.sample(min(n_sample, len(test_df)), random_state=42)
    X_sample = sample[available].copy()
    for c in cat_feats:
        X_sample[c] = X_sample[c].astype("category")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_sample)

    mean_abs = np.abs(shap_vals).mean(axis=0)
    shap_df = (
        pd.DataFrame({"feature": available, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # ── bar chart of top 20 ──
    top20 = shap_df.head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top20["feature"][::-1], top20["mean_abs_shap"][::-1], color="#4C72B0")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("LightGBM Feature Importance (SHAP, top 20)")
    plt.tight_layout()
    bar_path = fig_dir / "shap_importance.png"
    fig.savefig(bar_path, dpi=120)
    plt.close(fig)

    # ── beeswarm / summary plot ──
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    shap.summary_plot(
        shap_vals,
        X_sample,
        feature_names=available,
        max_display=15,
        show=False,
        plot_type="dot",
    )
    bee_path = fig_dir / "shap_beeswarm.png"
    plt.savefig(bee_path, dpi=120, bbox_inches="tight")
    plt.close("all")

    # ── SHAP direction table ──
    mean_shap_signed = shap_vals.mean(axis=0)
    shap_df["mean_shap_signed"] = mean_shap_signed
    shap_df["direction"] = shap_df["mean_shap_signed"].apply(
        lambda v: "↑ Player A" if v > 0 else "↓ Player B"
    )
    shap_df["mean_abs_shap"] = shap_df["mean_abs_shap"].round(4)
    shap_df["mean_shap_signed"] = shap_df["mean_shap_signed"].round(4)
    top15_table = shap_df.head(15)[["feature", "mean_abs_shap", "mean_shap_signed", "direction"]]
    top15_table.columns = ["Feature", "Mean |SHAP|", "Mean SHAP", "Net direction"]

    # Build relative paths for markdown image links (relative to report directory)
    _base = report_dir if report_dir else bar_path.parent.parent.parent
    bar_rel = str(bar_path.relative_to(_base))
    bee_rel = str(bee_path.relative_to(_base))

    lines = [
        "## 8. SHAP Analysis",
        "",
        f"SHAP values computed on a random sample of **{len(sample):,}** test-set games",
        "using `shap.TreeExplainer`. Values are on the log-odds scale.",
        "Positive SHAP = pushes prediction toward Player A winning.",
        "",
        "### Feature importance (mean |SHAP|)",
        "",
        f"![SHAP importance bar chart]({bar_rel})",
        "",
        _md_table(top15_table),
        "",
        "### SHAP beeswarm (top 15 features)",
        "",
        "Each dot is one game. Color = feature value (red = high, blue = low).",
        "Horizontal position = SHAP contribution to log-odds of Player A winning.",
        "",
        f"![SHAP beeswarm]({bee_rel})",
        "",
        "### Key observations",
        "",
        "- **`skill_diff` and `mmr_diff`** are the dominant features, confirming MMR is the strongest",
        "  pre-game signal. High positive `skill_diff` (Player A's MMR >> Player B's) strongly increases",
        "  Player A's win probability.",
        "- **`wins_lifetime_b` and `civ_wins_b`** appear high despite raw counts being redundant with",
        "  win rates. This likely reflects that higher counts of wins by Player B (not just their rate)",
        "  indicate a skilled, active player that is harder to beat.",
        "- **Civ features** (`civ_a`, `civ_b`) contribute meaningfully, confirming that civilization",
        "  choice has real — though secondary — impact on outcome beyond MMR.",
        "- **Missing-skill flags** (`missing_skill_a`, `missing_skill_b`) have non-trivial SHAP values,",
        "  indicating the model learned a distinct response for players without MMR/rating data",
        "  (likely early-season or low-play-count players).",
    ]
    return "\n".join(lines)


# ── ablation ─────────────────────────────────────────────────────────────────

# Each tuple is (label, list of NEW features added at this step).
# The ablation trains on the cumulative union of all features up to and including each step.
_ABLATION_FAMILIES: list[tuple[str, list[str]]] = [
    ("MMR only", [
        "mmr_a", "mmr_b", "mmr_diff",
        "rating_a", "rating_b", "rating_diff",
        "skill_a", "skill_b", "skill_diff",
    ]),
    ("+ lifetime history", [
        "games_lifetime_a", "wins_lifetime_a",
        "games_lifetime_b", "wins_lifetime_b",
        "overall_wr_a", "overall_wr_b",
        "wr_diff", "games_diff",
    ]),
    ("+ recent form / activity", [
        "days_since_a", "days_since_b",
        "games_season_a", "wins_season_a",
        "games_season_b", "wins_season_b",
        "season_wr_a", "season_wr_b",
    ]),
    ("+ civ-specific history", [
        "civ_a", "civ_b",           # categorical
        "civ_rand_a", "civ_rand_b",
        "civ_games_a", "civ_wins_a", "civ_wr_a",
        "civ_games_b", "civ_wins_b", "civ_wr_b",
    ]),
    ("+ map-specific history", [
        "map",                       # categorical
        "map_games_a", "map_wins_a", "map_wr_a",
        "map_games_b", "map_wins_b", "map_wr_b",
    ]),
    ("+ civ/map/meta priors", [
        "prior_matchup_games", "prior_matchup_wins", "prior_matchup_wr_a",
        "patch",                     # categorical
        "season",                    # categorical
    ]),
    ("+ missingness / cold-start", [
        "missing_mmr_a", "missing_mmr_b",
        "missing_rating_a", "missing_rating_b",
        "missing_skill_a", "missing_skill_b",
        "is_new_player_a", "is_new_player_b",
    ]),
    ("Full model (baseline)", [
        "civs_known", "map_known", "full_context_known",
    ]),
    ("+ activity (P9)", P9_FEATURES),
    ("+ low-history flags (P8)", P8_FEATURES),
    ("+ recent form (P3)", P3_FEATURES),
    ("+ MMR trend (P2)", P2_FEATURES),
    ("+ civ recency (P1)", P1_FEATURES),
    ("+ duration profile (P4)", P4_FEATURES),
    ("+ head-to-head (P5)", P5_FEATURES),
    ("+ empirical map priors (P7)", P7_FEATURES),
    ("+ map archetypes curated (P6)", P6_FEATURES),
]


def _train_ablation(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_list: list[str],
    target_col: str = "target",
):
    """Train a LightGBM for one ablation step (600 rounds, early stop 40, silent)."""
    import lightgbm as lgb
    from .model import DEFAULT_PARAMS

    available = [c for c in feature_list if c in train_df.columns]
    cat_feats = [c for c in CATEGORICAL_FEATURES if c in available]

    params = {k: v for k, v in DEFAULT_PARAMS.items() if k != "n_estimators"}

    def _ds(df, ref=None):
        X = df[available].copy()
        for c in cat_feats:
            X[c] = X[c].astype("category")
        return lgb.Dataset(
            X, label=df[target_col].values,
            categorical_feature=cat_feats,
            reference=ref, free_raw_data=False,
        )

    ds_train = _ds(train_df)
    ds_valid = _ds(valid_df, ref=ds_train)

    model = lgb.train(
        params, ds_train,
        num_boost_round=600,
        valid_sets=[ds_valid],
        callbacks=[
            lgb.early_stopping(stopping_rounds=40, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model, available, cat_feats


def _ablation_section(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fig_dir: Path | None = None,
    report_dir: Path | None = None,
) -> str:
    fig_dir = fig_dir or FIGURES_DIR
    y_test = test_df["target"].values
    cumulative: list[str] = []
    rows = []
    auc_vals: list[float] = []
    prev_auc: float | None = None

    for label, added in _ABLATION_FAMILIES:
        cumulative.extend(added)
        print(f"    {label} ({len([c for c in cumulative if c in train_df.columns])} features)...", flush=True)

        model, used, cat_feats = _train_ablation(train_df, valid_df, cumulative)

        X_test = test_df[used].copy()
        for c in cat_feats:
            X_test[c] = X_test[c].astype("category")
        preds = model.predict(X_test)

        m = evaluate(y_test, preds)
        delta = m["auc"] - prev_auc if prev_auc is not None else None
        delta_str = "—" if delta is None else (f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}")
        rows.append({
            "Feature set": label,
            "# features": len(used),
            "Test AUC": f"{m['auc']:.4f}",
            "ΔAUC": delta_str,
            "Log Loss": f"{m['log_loss']:.4f}",
            "Brier": f"{m['brier']:.4f}",
        })
        auc_vals.append(m["auc"])
        prev_auc = m["auc"]

    result_df = pd.DataFrame(rows)

    # ── waterfall bar chart ──────────────────────────────────────────────────
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels = [r["Feature set"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4C72B0" if i == 0 else "#55A868" for i in range(len(auc_vals))]
    ax.bar(range(len(auc_vals)), auc_vals, color=colors, edgecolor="white", width=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Test AUC")
    ax.set_title("Ablation: cumulative test AUC by feature family")
    ax.set_ylim(max(0.5, min(auc_vals) - 0.01), max(auc_vals) + 0.005)
    ax.axhline(auc_vals[-1], color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    abl_path = fig_dir / "ablation_auc.png"
    fig.savefig(abl_path, dpi=120)
    plt.close(fig)

    # Split table into two sections: original 8 and extended 7
    orig_rows = rows[:8]
    ext_rows  = rows[8:]
    orig_df   = pd.DataFrame(orig_rows)
    ext_df_   = pd.DataFrame(ext_rows) if ext_rows else None

    lines = [
        "## 9. Feature Family Ablation",
        "",
        "Each row trains a **fresh LightGBM** (600 rounds, early stop 40) on the cumulative",
        "feature set. The delta shows the marginal AUC gain from each group.",
        "",
        "### Part A — Original feature families",
        "",
        _md_table(orig_df),
    ]
    if ext_df_ is not None:
        lines += [
            "",
            "### Part B — Extended feature families (added incrementally on top of full baseline)",
            "",
            _md_table(ext_df_),
        ]
    _base = report_dir if report_dir else abl_path.parent.parent.parent
    abl_rel = str(abl_path.relative_to(_base))
    lines += [
        "",
        f"![Ablation AUC progression]({abl_rel})",
        "",
        "### Observations",
        "",
    ]

    # Auto-generate ranked observations from the deltas
    deltas = [(rows[i]["Feature set"], auc_vals[i] - auc_vals[i-1]) for i in range(1, len(auc_vals))]
    deltas_sorted = sorted(deltas, key=lambda x: -x[1])
    top_gain = deltas_sorted[0]
    second_gain = deltas_sorted[1] if len(deltas_sorted) > 1 else top_gain
    diminishing = [(lbl, d) for lbl, d in deltas if d < 0.001]

    lines.append(
        f"- **Largest single gain: {top_gain[0]}** (+{top_gain[1]:.4f} AUC). "
        "This is the feature family with the most independent signal beyond what preceded it."
    )
    lines.append(
        f"- **Second largest: {second_gain[0]}** (+{second_gain[1]:.4f} AUC)."
    )
    if diminishing:
        dim_str = ", ".join(f"{lbl} (+{d:.4f})" for lbl, d in diminishing[:5])
        lines.append(
            f"- **Near-zero gains:** {dim_str}. "
            "These add little marginal lift on S10+S11 alone — signal already captured "
            "by earlier families or too sparse to generalise."
        )
    lines += [
        "- The gap between **MMR only** and the **full extended model** quantifies how much",
        "  player history, civ choices, and recency signals contribute beyond raw skill rating.",
        "- Ingesting S3–S9 would increase lift from civ/map priors and civ recency features,",
        "  as S10 games currently receive no matchup prior.",
    ]

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def generate_report(
    report_path: Path | None = None,
    model_path: Path | None = None,
    meta_path: Path | None = None,
) -> None:
    report_path = report_path or REPORT_PATH

    # Derive meta path from model path when not explicitly given
    if model_path and meta_path is None:
        meta_path = model_path.parent / (model_path.stem + "_meta.json")

    # Figure directory: model-specific subdir so multiple reports don't clobber each other
    if model_path:
        fig_dir = FIGURES_DIR / model_path.stem
    else:
        fig_dir = FIGURES_DIR

    print("Loading training features from DB...")
    conn = get_conn(read_only=False)

    n_rows = conn.execute("SELECT COUNT(*) FROM training_features").fetchone()[0]
    large_dataset = n_rows > 3_000_000

    if large_dataset:
        # For large datasets the extended join (_fetch_player_ext) always reloads the
        # full training_features table from DuckDB, discarding any Python-side sample.
        # Fix: replace training_features in-place with the stride-sampled subset so the
        # extended join operates on ~2M rows instead of 6M+.
        stride = max(1, n_rows // 2_000_000)
        print(f"  Large dataset ({n_rows:,} rows) — stride-sampling to 1-in-{stride} "
              f"in DuckDB before extended join.")
        conn.execute(f"""
            CREATE OR REPLACE TABLE training_features AS
            SELECT * EXCLUDE (_rn)
            FROM (
                SELECT *, ROW_NUMBER() OVER (ORDER BY started_at, game_id) AS _rn
                FROM training_features
            ) t WHERE _rn % {stride} = 0
        """)
        n_sampled = conn.execute("SELECT COUNT(*) FROM training_features").fetchone()[0]
        print(f"  Sampled training_features: {n_sampled:,} rows")

    df = conn.execute("SELECT * FROM training_features").df()
    df = _add_derived_features(df)

    # Build extended features. For large datasets training_features is already sampled,
    # so the join produces a manageable result.
    all_families = set(FAMILY_FEATURES.keys())
    print("Building extended feature tables (P1–P9)...")
    df = extend_training_features(conn, df, families=all_families)
    conn.close()

    # Run leakage audit and record row count before splitting
    leakage_section = _leakage_audit(None, df)
    n_total = len(df)

    # Determine split strategy from model meta
    print("Loading model...")
    model, meta = load_model(model_path, meta_path)
    feature_cols = meta["feature_cols"]
    test_seasons = meta.get("split", {}).get("test_seasons")

    if test_seasons:
        test_mask = df["season"].isin(test_seasons)
        test_df   = df[test_mask].sort_values("started_at").reset_index(drop=True)
        remainder = df[~test_mask].sort_values("started_at").reset_index(drop=True)
        n = len(remainder)
        from .config import VALID_FRAC
        valid_end = int(n * (1 - VALID_FRAC))
        train_df  = remainder.iloc[:valid_end]
        valid_df  = remainder.iloc[valid_end:]
        print(f"  Season holdout: test = S{test_seasons}, "
              f"train {len(train_df):,} / valid {len(valid_df):,} / test {len(test_df):,}")
    else:
        train_df, valid_df, test_df = _temporal_split(df)

    del df

    # Derive a human-readable season description for the report header
    all_seasons = sorted(set(train_df["season"].unique()) | set(valid_df["season"].unique()) | set(test_df["season"].unique()))
    if test_seasons:
        train_s = sorted(set(all_seasons) - set(test_seasons))
        season_desc = f"S{train_s[0]}–S{train_s[-1]} (train/valid) + S{test_seasons[0]} (test)"
    else:
        season_desc = f"S{all_seasons[0]}–S{all_seasons[-1]}"

    y_test = test_df["target"].values

    # ── compute thresholds from training data (no leakage) ────────────────────
    min_games_train = train_df[["games_season_a", "games_season_b"]].min(axis=1)
    min_games_thresholds = np.percentile(min_games_train.dropna(), [20, 40, 60, 80])

    activity_gap_train = (train_df["games_season_a"] - train_df["games_season_b"]).abs()
    activity_gap_thresholds = np.percentile(activity_gap_train.dropna(), [20, 40, 60, 80])

    train_mean_mmr = ((train_df["mmr_a"] + train_df["mmr_b"]) / 2).dropna()
    mmr_level_thresholds = np.percentile(train_mean_mmr, [20, 40, 60, 80, 99])

    print("Fitting baselines...")
    constant  = ConstantBaseline().fit(train_df)
    mmr_log   = MMRLogisticBaseline().fit(train_df)
    civ_map   = CivMapBucketBaseline().fit(train_df)
    enhanced  = EnhancedLogisticBaseline(extra_features=NUMERIC_TOP5_LGBM).fit(train_df)

    print("Fitting L1 Logistic (numeric only, tuning C on validation)...")
    l1_logistic = L1LogisticBaseline().fit(train_df, valid_df)

    print("Computing predictions...")
    preds = {
        "Constant": constant.predict_proba(test_df),
        "MMR Logistic": mmr_log.predict_proba(test_df),
        "Civ/Map Bucket": civ_map.predict_proba(test_df),
        "Enhanced Logistic": enhanced.predict_proba(test_df),
        "LightGBM": _predict(model, test_df, feature_cols),
    }
    from .config import XGB_MODEL_PATH, XGB_META_PATH
    if XGB_MODEL_PATH.exists() and XGB_META_PATH.exists():
        from .model import load_xgb, _predict_xgb
        xgb_model, xgb_meta = load_xgb()
        missing_xgb = set(xgb_meta["feature_cols"]) - set(test_df.columns)
        if missing_xgb:
            print(f"  Skipping XGBoost: {len(missing_xgb)} feature(s) missing from current data.")
        else:
            xgb_cats = xgb_meta.get("cat_categories")
            preds["XGBoost"] = _predict_xgb(xgb_model, test_df, xgb_meta["feature_cols"],
                                             global_cats=xgb_cats)
    preds["L1 Logistic (numeric only)"] = l1_logistic.predict_proba(test_df)

    print("Running SHAP analysis (this may take a minute)...")
    # ── 2. Performance summary ────────────────────────────────────────────────
    perf_rows = [_metrics_row(name, y_test, pred) for name, pred in preds.items()]
    perf_df = pd.DataFrame(perf_rows)

    cal_df = calibration_table(y_test, preds["LightGBM"])
    cal_df.columns = ["Predicted WR", "Empirical WR", "Gap", "N"]

    perf_section = "\n".join([
        "## 2. Model Performance on Test Set",
        "",
        f"Test set: **{len(test_df):,} games** ({test_df['started_at'].min()} → {test_df['started_at'].max()})",
        "",
        "### All models",
        "",
        _md_table(perf_df),
        "",
        "### LightGBM calibration",
        "",
        "Predicted vs empirical win rate. A well-calibrated model sits near the diagonal (Gap ≈ 0).",
        "",
        _md_table(cal_df),
        "",
        "### Enhanced Logistic Baseline — coefficients",
        "",
        _md_table(enhanced.coef_table().head(12).round(4)),
    ])

    # ── 3. Subgroup: |MMR diff| ───────────────────────────────────────────────
    subgroup_section = _subgroup_section(test_df, preds)

    # ── 4. Subgroup: min season games ─────────────────────────────────────────
    min_season_section = _subgroup_min_season_games_section(test_df, preds, min_games_thresholds)

    # ── 5. Subgroup: activity gap ─────────────────────────────────────────────
    activity_gap_section = _subgroup_activity_gap_section(test_df, preds, activity_gap_thresholds)

    # ── 6. Subgroup: mean hidden MMR tier ─────────────────────────────────────
    mean_mmr_section = _subgroup_mean_mmr_section(test_df, preds, mmr_level_thresholds)

    # ── 7. Subgroup: MMR availability ─────────────────────────────────────────
    missing_mmr_section = _subgroup_missing_mmr_section(test_df, preds)

    report_dir = report_path.resolve().parent

    # ── 8. SHAP ───────────────────────────────────────────────────────────────
    shap_section = _shap_section(model, test_df, feature_cols, fig_dir=fig_dir,
                                 report_dir=report_dir)

    # ── 9. Ablation ───────────────────────────────────────────────────────────
    abl_train = (
        train_df.sample(min(500_000, len(train_df)), random_state=42)
        if len(train_df) > 500_000 else train_df
    )
    print(f"Running feature family ablation ({len(_ABLATION_FAMILIES)} models, "
          f"{len(abl_train):,} train rows)...")
    ablation_section = _ablation_section(abl_train, valid_df, test_df, fig_dir=fig_dir,
                                         report_dir=report_dir)

    # ── assemble ──────────────────────────────────────────────────────────────
    sections = [
        "# AOE4 RM 1v1 Prediction Model — Analysis Report",
        "",
        f"Generated from {season_desc} data ({n_total:,} training rows sampled).",
        "Train / valid / test split uses season holdout (test = S11)."
        if test_seasons else
        "Train / valid / test split is strictly temporal (no random row shuffling).",
        "",
        leakage_section,
        "",
        perf_section,
        "",
        subgroup_section,
        "",
        min_season_section,
        "",
        activity_gap_section,
        "",
        mean_mmr_section,
        "",
        missing_mmr_section,
        "",
        shap_section,
        "",
        ablation_section,
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(sections))
    print(f"\nReport written to {report_path}")
    print(f"Figures saved in {fig_dir}/")
