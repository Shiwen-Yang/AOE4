"""Generate reports/generated/civ_choice_report.md."""
from pathlib import Path

import numpy as np

from .features import ALL_FEATURES

BASE_DIR = Path(__file__).resolve().parents[2]
REPORT_PATH = BASE_DIR / "reports" / "generated" / "civ_choice_report.md"


def _fmt(m: dict) -> str:
    if not m or m.get("n", 0) == 0:
        return "N/A"
    return (
        f"N={m['n']:,}  Top1={m['top1_acc']:.3f}  Top3={m['top3_acc']:.3f}  "
        f"Top5={m['top5_acc']:.3f}  LogLoss={m['log_loss']:.4f}  "
        f"Brier={m['brier']:.4f}  MeanP={m['mean_chosen_prob']:.4f}"
    )


def _metrics_table(metrics_dict: dict[str, dict]) -> str:
    lines = [
        "| Model | N | Top-1 | Top-3 | Top-5 | LogLoss | Brier | MeanP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in metrics_dict.items():
        if not m or m.get("n", 0) == 0:
            lines.append(f"| {name} | — | — | — | — | — | — | — |")
        else:
            lines.append(
                f"| {name} | {m['n']:,} | {m['top1_acc']:.3f} | {m['top3_acc']:.3f} | "
                f"{m['top5_acc']:.3f} | {m['log_loss']:.4f} | {m['brier']:.4f} | "
                f"{m['mean_chosen_prob']:.4f} |"
            )
    return "\n".join(lines)


def _subgroup_table(subgroup_metrics: dict[str, dict]) -> str:
    lines = [
        "| Subgroup | N | Top-1 | Top-3 | LogLoss |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, m in subgroup_metrics.items():
        if not m or m.get("n", 0) == 0:
            lines.append(f"| {label} | — | — | — | — |")
        else:
            lines.append(
                f"| {label} | {m['n']:,} | {m['top1_acc']:.3f} | "
                f"{m['top3_acc']:.3f} | {m['log_loss']:.4f} |"
            )
    return "\n".join(lines)


def _shap_table(mean_abs: np.ndarray, feature_names: list, top_n: int = 15) -> str:
    if len(mean_abs) == 0:
        return "_SHAP not computed._"
    idx = np.argsort(mean_abs)[::-1][:top_n]
    lines = ["| Rank | Feature | Mean |SHAP| |", "|---:|---|---:|"]
    for rank, i in enumerate(idx, 1):
        lines.append(f"| {rank} | `{feature_names[i]}` | {mean_abs[i]:.4f} |")
    return "\n".join(lines)


def _findings(
    val_stats: dict,
    all_metrics: dict[str, dict],
    subgroup_metrics: dict[str, dict],
    shap_mean: np.ndarray,
    feature_names: list,
    normalization_winner: str,
) -> str:
    findings = []

    lgbm = all_metrics.get("LightGBM", {})
    last = all_metrics.get("Last-civ", {})
    if lgbm.get("n", 0) > 0:
        top1 = lgbm["top1_acc"]
        if top1 > 0.65:
            findings.append(
                f"LightGBM achieves Top-1 accuracy of {top1:.1%} overall. This is inflated "
                "by trivial rejection of civs the player has never played."
            )
        else:
            findings.append(
                f"LightGBM achieves Top-1 accuracy of {top1:.1%} overall."
            )

    played_only = subgroup_metrics.get("Played-civ rows only", {})
    if played_only.get("n", 0) > 0:
        findings.append(
            f"Among candidate civs the player has actually used "
            f"(played-civ-only), Top-1={played_only['top1_acc']:.1%}. "
            "This is the meaningful measure of civ-switching prediction quality."
        )

    if lgbm.get("n", 0) > 0 and last.get("n", 0) > 0:
        delta = lgbm["top1_acc"] - last["top1_acc"]
        if abs(delta) < 0.02:
            findings.append("LightGBM roughly ties the last-civ baseline — the strongest predictor is simple recency.")
        elif delta > 0:
            findings.append(
                f"LightGBM improves {delta:+.1%} over last-civ baseline, capturing map/patch switching signals."
            )
        else:
            findings.append(
                f"Last-civ baseline outperforms LightGBM by {-delta:.1%}. Model may need more expressive features."
            )

    findings.append(
        f"Normalization method '{normalization_winner}' gave lower validation log loss."
    )

    if len(shap_mean) > 0:
        top2 = [feature_names[i] for i in np.argsort(shap_mean)[::-1][:2]]
        findings.append(
            f"Top SHAP features: `{top2[0]}` and `{top2[1]}`. "
            "If these are civ-frequency features, the model is mostly learning player habits."
        )

    return "\n".join(f"- {f}" for f in findings)


def generate_report(
    seasons: list[int],
    val_stats: dict,
    randomized_pct: float,
    all_metrics: dict[str, dict],
    subgroup_metrics: dict[str, dict],
    shap_mean: np.ndarray,
    feature_names: list,
    normalization_winner: str,
) -> str:
    findings = _findings(
        val_stats, all_metrics, subgroup_metrics, shap_mean, feature_names, normalization_winner
    )

    return f"""# AOE4 Civilization-Choice Prediction (V1)

## 1. Objective

Predict which civilization a player will choose before an AOE4 RM 1v1 match,
using only pre-match information (player history, map, patch, MMR/rating).
Enables win-probability marginalization when civilizations are unknown.

Seasons: {seasons}

---

## 2. Dataset Construction

One row per `(game_id, profile_id, candidate_civ)`.
Target = 1 for the actually chosen civ, 0 for all others.

| Metric | Value |
|---|---|
| Total candidate rows | {val_stats['n_total']:,} |
| Player-match groups | {val_stats['n_groups']:,} |
| Avg candidate civs per group | {val_stats['n_civs_per_group']:.1f} |

---

## 3. Randomized-Civ Filtering

Randomized/null civilization rows removed: **{randomized_pct:.1f}%** of participant rows.

These rows are excluded from V1. Future work: two-stage model (predict whether player
chose random, then predict civ if not).

---

## 4. Candidate Set

Candidate civs for each game are limited to civilizations first seen at or before
the game's start timestamp (`MIN(started_at)` per civ from data). For S10+S11 all
18 civs are valid in every game.

---

## 5. Feature Summary

**Candidate features**: {len(ALL_FEATURES)} model features including pick shares
(lifetime, 30d, patch, map), win rates, exact recent-pick rank flags, global pick rates, and civ ranks.

**Player features**: MMR/rating, game counts, civ-pool entropy, main-civ share.

**Context**: map, patch, season (categorical, LightGBM native).

---

## 6. Baseline Performance (test set)

{_metrics_table({k: v for k, v in all_metrics.items() if k != "LightGBM"})}

---

## 7. LightGBM Performance (test set)

Normalization method used: **{normalization_winner}** (lower validation log loss).

{_metrics_table({"LightGBM": all_metrics.get("LightGBM", {})})}

---

## 8. Subgroup Analysis

{_subgroup_table(subgroup_metrics)}

---

## 9. Played-Civ-Only Evaluation

Restricts evaluation to candidate rows where `cand_games_lifetime > 0` (player has
used this civ before). This answers: *among civs this player realistically plays,
can we rank them correctly?*

{_fmt(subgroup_metrics.get("Played-civ rows only", {}))}

---

## 10. SHAP / Feature Importance

{_shap_table(shap_mean, feature_names)}

---

## 11. Main Findings

{findings}

---

## 12. Integration with Win-Probability Model

`predict_with_civ_marginalization(player_a, player_b, map)` computes:

```
P(A wins) = Σ_civA Σ_civB  P(civA|A,map) × P(civB|B,map) × P(A wins|A,B,map,civA,civB)
```

All 18×18=324 win-model evaluations are batched. See `civ_choice/integrate.py`.

---

## 13. Limitations

- 20% game-level training sample; final metrics on full test set.
- Randomized civ choices excluded from training (V1).
- 30d, patch, map, and exact last-20-games stats via LATERAL aggregations — exact but computed at training time only.
- `candidate_was_played_last_3/5_games` remains approximated from 30d count, but last-20-games top ranks are exact.
- Global pick rate this-season approximates with full-season rate (includes future games in season).

---

## 14. Next Steps

- Model random-civ behavior as a two-stage problem.
- Add `games_since_last_played_civ` (count-based, not day-based).
- Add `LGBMRanker` with `lambdarank` objective as a comparison.
- Calibrate probabilities with temperature scaling.
- Retrain on all seasons for production inference.
"""


def write_report(content: str, path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"\n  Report written to {path}")
