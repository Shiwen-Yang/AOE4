"""Generate reports/generated/rating_update_report.md."""
from pathlib import Path
from typing import Any

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
REPORT_PATH = BASE_DIR / "reports" / "generated" / "rating_update_report.md"


def _fmt_metrics(m: dict, indent: str = "") -> str:
    if m.get("n", 0) == 0:
        return f"{indent}N/A (no valid rows)"
    return (
        f"{indent}N={m['n']:,}  "
        f"MAE={m['mae']:.3f}  "
        f"RMSE={m['rmse']:.3f}  "
        f"R²={m['r2']:.4f}  "
        f"MeanErr={m['mse_signed']:.3f}  "
        f"MedAE={m['medae']:.3f}"
    )


def _metrics_table(metrics_dict: dict[str, dict]) -> str:
    lines = [
        "| Model / Baseline | N | MAE | RMSE | R² | Mean Err | MedAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in metrics_dict.items():
        if m.get("n", 0) == 0:
            lines.append(f"| {name} | — | — | — | — | — | — |")
        else:
            lines.append(
                f"| {name} | {m['n']:,} | {m['mae']:.3f} | {m['rmse']:.3f} | "
                f"{m['r2']:.4f} | {m['mse_signed']:.3f} | {m['medae']:.3f} |"
            )
    return "\n".join(lines)


def _subgroup_table(subgroup_metrics: dict[str, dict]) -> str:
    lines = [
        "| Subgroup | N | MAE | RMSE | R² |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, m in subgroup_metrics.items():
        if m.get("n", 0) == 0:
            lines.append(f"| {label} | — | — | — | — |")
        else:
            lines.append(
                f"| {label} | {m['n']:,} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['r2']:.4f} |"
            )
    return "\n".join(lines)


def _shap_table(mean_abs_shap: np.ndarray, feature_names: list[str], top_n: int = 15) -> str:
    if len(mean_abs_shap) == 0:
        return "_SHAP not available (shap package not installed)._"
    idx = np.argsort(mean_abs_shap)[::-1][:top_n]
    lines = [
        "| Rank | Feature | Mean |SHAP| |",
        "|---:|---|---:|",
    ]
    for rank, i in enumerate(idx, 1):
        lines.append(f"| {rank} | `{feature_names[i]}` | {mean_abs_shap[i]:.4f} |")
    return "\n".join(lines)


def _k_seg_table(k_segments: dict) -> str:
    if not k_segments:
        return "_No K-factor segmentation data._"
    lines = [
        "| Segment | N | Best K | MAE | R² |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, v in k_segments.items():
        lines.append(
            f"| {label} | {v['n']:,} | {v['K']:.1f} | {v['MAE']:.3f} | {v['R2']:.4f} |"
        )
    return "\n".join(lines)


def _build_findings(
    val_stats: dict,
    det_result: dict,
    elo_K: float,
    elo_D: float,
    elo_mae: float,
    k_segments: dict,
    all_metrics: dict[str, dict],
    residual_stats: dict,
) -> str:
    findings = []

    pct = det_result.get("pct_single_delta", 0)
    if pct > 95:
        findings.append(
            f"The visible rating update is **nearly deterministic** given "
            f"(player_rating, opponent_rating, result): {pct:.1f}% of unique input "
            f"tuples map to exactly one delta value."
        )
    else:
        findings.append(
            f"The update is **not fully deterministic** in (player_rating, opponent_rating, result): "
            f"only {pct:.1f}% of tuples map to a single delta, suggesting additional hidden inputs."
        )

    if elo_mae < 2:
        findings.append(
            f"A standard Elo formula with K={elo_K:.1f}, D={elo_D:.0f} fits the data very well "
            f"(MAE={elo_mae:.3f}). The rating update is essentially Elo-based."
        )
    elif elo_mae < 5:
        findings.append(
            f"An Elo formula (K={elo_K:.1f}, D={elo_D:.0f}) partially fits (MAE={elo_mae:.3f}), "
            f"indicating a modified Elo or additional adjustment factors."
        )
    else:
        findings.append(
            f"Standard Elo (best K={elo_K:.1f}, D={elo_D:.0f}) does not fit well (MAE={elo_mae:.3f}). "
            f"The rating system may not be Elo-based."
        )

    if k_segments:
        ks = [f"{label.split('(')[0].strip()}: K={v['K']:.1f}" for label, v in k_segments.items()]
        findings.append(
            f"K-factor varies by player experience — {'; '.join(ks)}. "
            f"Higher K for new players is consistent with dynamic K (like FIDE's system)."
        )

    lgbm_m = all_metrics.get("LightGBM", {})
    elo_m = all_metrics.get("Elo", {})
    if lgbm_m.get("n", 0) > 0:
        r2_lgbm = lgbm_m["r2"]
        r2_elo = elo_m.get("r2", 0)
        if abs(r2_lgbm - r2_elo) < 0.01:
            findings.append(
                f"LightGBM (R²={r2_lgbm:.4f}) adds little over the Elo baseline (R²={r2_elo:.4f}), "
                f"confirming that rating_gap and result already explain most variance."
            )
        else:
            findings.append(
                f"LightGBM (R²={r2_lgbm:.4f}) outperforms the Elo baseline (R²={r2_elo:.4f}), "
                f"suggesting additional structure beyond the Elo formula."
            )

    res_corr = residual_stats.get("correlations", {})
    mmr_corr = res_corr.get("player_mmr_before", float("nan"))
    gs_corr = res_corr.get("games_this_season_before", float("nan"))
    if not np.isnan(mmr_corr) and abs(mmr_corr) > 0.05:
        findings.append(
            f"Elo residuals correlate with hidden MMR (r={mmr_corr:.3f}), suggesting the rating "
            f"update may incorporate hidden MMR as an additional factor."
        )
    if not np.isnan(gs_corr) and abs(gs_corr) > 0.05:
        findings.append(
            f"Elo residuals correlate with games_this_season (r={gs_corr:.3f}), consistent with "
            f"a dynamic K or provisional-period adjustment."
        )

    return "\n".join(f"- {f}" for f in findings)


def generate_report(
    seasons: list[int],
    val_stats: dict,
    det_result: dict,
    elo_K: float,
    elo_D: float,
    elo_mae: float,
    k_segments: dict,
    residual_stats: dict,
    all_metrics: dict[str, dict],
    subgroup_metrics: dict[str, dict],
    shap_mean_abs: np.ndarray,
    feature_names: list[str],
) -> str:
    findings = _build_findings(
        val_stats, det_result, elo_K, elo_D, elo_mae,
        k_segments, all_metrics, residual_stats,
    )
    _corr_rows = "\n".join(
        f"| `{k}` | {v:.4f} |"
        for k, v in residual_stats.get("correlations", {}).items()
    )

    return f"""# AOE4 Visible Rating Update Investigation

## 1. Objective

Understand how the visible rating (`rating_diff`) is updated after each AOE4 RM 1v1 match.
Primary question: **Is the update deterministic and Elo-based?**

Seasons analysed: {seasons}

---

## 2. Data Construction

One row per participant per match. Both players from the same game appear as separate rows.
Temporal features (games before, streak, recent win rate) are computed over full multi-season history.

| Metric | Value |
|---|---|
| Total rows | {val_stats['total_rows']:,} |
| rating_diff present | {val_stats['rating_delta_pct']:.1f}% |
| mmr_diff present (secondary) | {val_stats['mmr_delta_pct']:.1f}% |
| Winners with positive delta | {val_stats['winner_positive_pct']:.1f}% |
| Losers with negative delta | {val_stats['loser_negative_pct']:.1f}% |
| Winners with non-positive delta | {val_stats['winner_nonpos_count']:,} |
| Losers with non-negative delta | {val_stats['loser_nonneg_count']:,} |
| Mean delta — winners | {val_stats['mean_delta_winner']:.2f} |
| Mean delta — losers | {val_stats['mean_delta_loser']:.2f} |
| Median delta — winners | {val_stats['median_delta_winner']:.2f} |
| Median delta — losers | {val_stats['median_delta_loser']:.2f} |
| Delta range [min, max] | [{val_stats['delta_min']:.0f}, {val_stats['delta_max']:.0f}] |
| Percentiles [p5, p25, p50, p75, p95] | [{val_stats['delta_p5']:.0f}, {val_stats['delta_p25']:.0f}, {val_stats['delta_p50']:.0f}, {val_stats['delta_p75']:.0f}, {val_stats['delta_p95']:.0f}] |

---

## 3. Formula Recovery

### 3a. Determinism Check

Grouped by `(player_rating_before, opponent_rating_before, result)`.

| | Value |
|---|---|
| Unique input tuples | {det_result['n_unique_inputs']:,} |
| Tuples with exactly one delta | {det_result['n_single_delta']:,} ({det_result['pct_single_delta']:.1f}%) |
| Avg distinct deltas per tuple | {det_result['avg_distinct_deltas']:.2f} |

### 3b. Elo Formula Fit

Formula: `delta = K × (result − 1 / (1 + 10^((rating_b − rating_a) / D)))`

Grid search over K ∈ {{8..64}} × D ∈ {{100..700, step 25}}.

**Best fit: K = {elo_K:.1f}, D = {elo_D:.0f}**

Performance on full dataset (rows with both ratings non-null):

{_metrics_table({"Elo (best K, D)": all_metrics.get("Elo_full", {})})}

### 3c. K-Factor Segmentation (D = {elo_D:.0f})

{_k_seg_table(k_segments)}

### 3d. Residual Analysis

| Metric | Value |
|---|---|
| Residual mean | {residual_stats.get('mean', float('nan')):.4f} |
| Residual std | {residual_stats.get('std', float('nan')):.4f} |

Pearson correlations of Elo residual with:

| Variable | Correlation |
|---|---|
{_corr_rows}

---

## 4. Baseline Performance (test set)

{_metrics_table({k: v for k, v in all_metrics.items() if k not in ("LightGBM",)})}

---

## 5. LightGBM Performance (test set)

{_metrics_table({"LightGBM": all_metrics.get("LightGBM", {})})}

---

## 6. Subgroup Performance (LightGBM, test set)

{_subgroup_table(subgroup_metrics)}

---

## 7. Feature Importance (mean |SHAP|)

{_shap_table(shap_mean_abs, feature_names)}

---

## 8. Main Findings

{findings}

---

## 9. Limitations

- Analysis uses seasons {seasons} only; the rating formula may differ in earlier seasons.
- The Elo fit uses exact integer ratings; any hidden rounding in the game engine could inflate residuals.
- Hidden MMR (`mmr`) is absent for ~{100 - val_stats['mmr_delta_pct']:.0f}% of rows and cannot be included fully.
- Two participant rows per game are correlated; standard error estimates undercount true uncertainty.
- The temporal split prevents future leakage but very early-season rows have sparse history features.

---

## 10. Next Steps

- If the Elo formula fits well (MAE < 2): verify exact integer rounding behaviour in the game client.
- If K varies significantly by experience: model K as a function of `games_this_season_before` (dynamic K).
- If residuals correlate with hidden MMR: investigate whether the game uses a dual MMR+ELO system.
- If LightGBM substantially outperforms Elo: analyse top SHAP features to identify missing formula inputs.
- Add `predict_rating_delta(player_a, player_b, result)` as a product feature once the formula is understood.
"""


def write_report(content: str, path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"\n  Report written to {path}")
