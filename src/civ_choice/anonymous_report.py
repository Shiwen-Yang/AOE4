"""Report generation for anonymous opponent civ-choice prediction."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .anonymous_features import ANONYMOUS_FEATURES

BASE_DIR = Path(__file__).resolve().parents[2]
REPORT_PATH = BASE_DIR / "reports" / "generated" / "anonymous_opponent_civ_choice_report.md"


def _metrics_table(metrics: dict[str, dict]) -> str:
    lines = [
        "| Model | N | Top-1 | Top-3 | Top-5 | LogLoss | Brier | MeanP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in metrics.items():
        if not m:
            lines.append(f"| {name} | - | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {name} | {m['n']:,} | {m['top1_acc']:.3f} | {m['top3_acc']:.3f} | "
            f"{m['top5_acc']:.3f} | {m['log_loss']:.4f} | {m['brier']:.4f} | "
            f"{m['mean_chosen_prob']:.4f} |"
        )
    return "\n".join(lines)


def _shap_table(mean_abs: np.ndarray, feature_names: list[str], top_n: int = 15) -> str:
    if len(mean_abs) == 0:
        return "_SHAP not computed._"
    idx = np.argsort(mean_abs)[::-1][:top_n]
    lines = ["| Rank | Feature | Mean |SHAP| |", "|---:|---|---:|"]
    for rank, i in enumerate(idx, 1):
        lines.append(f"| {rank} | `{feature_names[i]}` | {mean_abs[i]:.4f} |")
    return "\n".join(lines)


def generate_anonymous_report(
    *,
    seasons: list[int],
    user_profile_id: int | None,
    val_stats: dict,
    all_metrics: dict[str, dict],
    subgroup_metrics: dict[str, dict],
    shap_mean: np.ndarray,
    feature_names: list[str],
    normalization_winner: str,
) -> str:
    mmr = all_metrics.get("MMR-tier pick-rate", {})
    lgbm = all_metrics.get("Anonymous LightGBM", {})
    delta = ""
    if mmr and lgbm:
        delta = (
            f"\n- LightGBM log-loss delta vs MMR-tier baseline: "
            f"{lgbm['log_loss'] - mmr['log_loss']:+.4f}."
        )

    user_mode = (
        f"Fixed user profile: `{user_profile_id}`."
        if user_profile_id is not None
        else "Generic anonymous mode: no fixed user recent-opponent features."
    )

    subgroup = _metrics_table(subgroup_metrics) if subgroup_metrics else "_No subgroup metrics._"

    return f"""# Anonymous Opponent Civ-Choice Prediction

## Objective

Predict an opponent's civilization without using the opponent profile history.
Allowed information is anonymous live-safe context: map, patch/season,
MMR/rating tier, and optional known-user recent-opponent meta.

{user_mode}

Seasons: {seasons}

## Dataset

| Metric | Value |
|---|---:|
| Candidate rows | {val_stats['n_total']:,} |
| Player-match groups | {val_stats['n_groups']:,} |
| Avg candidate civs per group | {val_stats['n_civs_per_group']:.1f} |

Feature count: {len(ANONYMOUS_FEATURES)}

## Test Metrics

{_metrics_table(all_metrics)}

## Subgroups

{subgroup}

## Feature Importance

{_shap_table(shap_mean, feature_names)}

## Findings

- Primary benchmark is `MMR-tier pick-rate`.
- Prediction normalization selected on validation: `{normalization_winner}`.{delta}
"""


def write_report(content: str, path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"  Wrote report: {path}")
