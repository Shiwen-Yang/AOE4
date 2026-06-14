"""
Feature pruning analysis for AOE4 RM 1v1 outcome prediction.

Phase A (no retraining):
  - Load the existing trained model
  - Rebuild the full feature matrix (training_features + all active families)
  - Compute SHAP on the temporal test split
  - Report per-family feature importance rankings + pruning candidates

Phase B (5 targeted training runs):
  - Compare full model against:
      core_P1P3P4      : P1 + P3 + P4 only (all marginal families dropped)
      drop_mmr_trend   : all families minus P2
      drop_head_to_head: all families minus P5
      drop_low_history : all families minus P8
      drop_activity    : all families minus P9
  - Hard assertions catch any silent feature drops before and after training
"""
import json
import tempfile
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from aoe4_predict.config import MODEL_META_PATH
from aoe4_predict.db import get_conn
from aoe4_predict.evaluate import evaluate
from aoe4_predict.features_extra import (
    DISABLED_FAMILIES,
    FAMILY_FEATURES,
    extend_training_features,
)
from aoe4_predict.model import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    DEFAULT_PARAMS,
    _temporal_split,
    load_model,
    train,
)

SHAP_SAMPLE = 5_000
PRUNE_THRESHOLD_FRAC = 0.01  # flag feature if mean|SHAP| < 1% of family top

ALL_ACTIVE = set(FAMILY_FEATURES) - DISABLED_FAMILIES
# Families whose significance was established in prior reports (not re-tested)
CORE_FAMILIES = {"civ_recency", "adjusted_form", "duration_profile"}
MARGINAL_FAMILIES = ALL_ACTIVE - CORE_FAMILIES  # P2, P5, P8, P9

ABLATION_CONFIGS: dict[str, set[str]] = {
    "core_P1P3P4":       CORE_FAMILIES,
    "drop_mmr_trend":    ALL_ACTIVE - {"mmr_trend"},
    "drop_head_to_head": ALL_ACTIVE - {"head_to_head"},
    "drop_low_history":  ALL_ACTIVE - {"low_history_detail"},
    "drop_activity":     ALL_ACTIVE - {"activity_session"},
}


# ── Validation helpers ────────────────────────────────────────────────────────

def _check_df_for_families(df: pd.DataFrame, families: set[str], label: str) -> None:
    """Raise ValueError on any silent feature drop or all-NaN column."""
    for fam in families:
        for feat in FAMILY_FEATURES[fam]:
            if feat not in df.columns:
                raise ValueError(
                    f"[{label}] Family '{fam}': feature '{feat}' missing after "
                    f"extend_training_features — check SQL in features_extra.py"
                )
            if df[feat].isna().all():
                raise ValueError(
                    f"[{label}] Feature '{feat}' (family '{fam}') is entirely NaN"
                )
            nan_frac = float(df[feat].isna().mean())
            if nan_frac > 0.5:
                print(
                    f"  WARNING [{label}]: '{feat}' has {nan_frac:.0%} NaN — "
                    "check query logic"
                )


def _check_model_has_families(
    meta: dict, families: set[str], label: str
) -> None:
    """Raise ValueError if any expected feature was silently dropped by the model."""
    model_cols = set(meta["feature_cols"])
    for fam in families:
        for feat in FAMILY_FEATURES[fam]:
            if feat not in model_cols:
                raise ValueError(
                    f"[{label}] Feature '{feat}' (family '{fam}') is in the DataFrame "
                    f"but absent from model feature_cols — add it to ALL_FEATURES in model.py"
                )


def _check_metrics_valid(meta: dict, label: str) -> None:
    m = meta["metrics"]["test"]
    if np.isnan(m["auc"]):
        raise ValueError(
            f"[{label}] Test AUC is NaN — test set may be empty or single-class. "
            f"Test rows: {meta['split']['test_rows']}"
        )
    if meta["split"]["test_rows"] < 5_000:
        raise ValueError(
            f"[{label}] Test set suspiciously small: {meta['split']['test_rows']} rows"
        )


# ── Phase A: SHAP on the existing model ──────────────────────────────────────

def run_shap_phase(conn) -> list[str]:
    print("\n── Phase A: SHAP analysis on existing model ─────────────────────────")

    model, meta = load_model()
    feature_cols: list[str] = meta["feature_cols"]
    print(f"  Model: {meta['n_trees']} trees, {len(feature_cols)} features")

    print(f"  Rebuilding full feature matrix ({len(ALL_ACTIVE)} families)...")
    df_full = extend_training_features(conn, None, ALL_ACTIVE)
    print(f"  DataFrame shape: {df_full.shape}")

    # 1. Every feature the model expects must be present
    missing = [f for f in feature_cols if f not in df_full.columns]
    if missing:
        raise ValueError(
            f"[SHAP] Model expects {len(missing)} features absent from df:\n  {missing}"
        )

    # 2. No all-NaN model feature
    all_nan = [f for f in feature_cols if df_full[f].isna().all()]
    if all_nan:
        raise ValueError(
            f"[SHAP] {len(all_nan)} model features are entirely NaN: {all_nan}"
        )

    # 3. Test split is substantial
    _, _, test_df = _temporal_split(df_full)
    if len(test_df) < 5_000:
        raise ValueError(f"[SHAP] Test split too small: {len(test_df)} rows")

    X_test = test_df[feature_cols].head(SHAP_SAMPLE).copy()
    cat_feats = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    for c in cat_feats:
        X_test[c] = X_test[c].astype("category")

    print(f"  Computing SHAP on {len(X_test):,} test rows...", flush=True)
    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    shap_arr = explainer.shap_values(X_test)
    elapsed = time.time() - t0
    print(f"  SHAP done in {elapsed:.1f}s")

    # 4. Shape assertion
    expected = (len(X_test), len(feature_cols))
    if shap_arr.shape != expected:
        raise ValueError(
            f"[SHAP] Shape mismatch: got {shap_arr.shape}, expected {expected}"
        )

    mean_abs = pd.Series(np.abs(shap_arr).mean(axis=0), index=feature_cols)

    family_order = [
        "civ_recency", "adjusted_form", "duration_profile",
        "mmr_trend", "head_to_head", "low_history_detail", "activity_session",
    ]

    lines: list[str] = []
    lines.append("## Per-Family SHAP Feature Rankings\n\n")
    lines.append(
        f"Computed on {len(X_test):,} test-set rows ({elapsed:.0f}s). "
        f"Pruning threshold: mean |SHAP| < {PRUNE_THRESHOLD_FRAC:.0%} of the "
        f"top feature in the family.\n\n"
    )

    for fam in family_order:
        feats = [f for f in FAMILY_FEATURES[fam] if f in mean_abs.index]
        if not feats:
            print(f"  [{fam}] No features found in model — skipping")
            continue
        ranked = mean_abs[feats].sort_values(ascending=False)
        top_val = float(ranked.iloc[0])
        threshold = top_val * PRUNE_THRESHOLD_FRAC
        candidates = [f for f, v in ranked.items() if v < threshold]

        print(f"\n  {fam} ({len(feats)} features, top={top_val:.5f}):")
        for feat, val in ranked.items():
            marker = " ← prune?" if feat in candidates else ""
            print(f"    {feat:<45} {val:.5f}{marker}")

        lines.append(f"### {fam} ({len(feats)} features)\n\n")
        lines.append("| Rank | Feature | Mean |SHAP| | Prune? |\n")
        lines.append("|---|---|---|---|\n")
        for i, (feat, val) in enumerate(ranked.items(), 1):
            prune = "yes" if feat in candidates else ""
            lines.append(f"| {i} | `{feat}` | {val:.5f} | {prune} |\n")
        if candidates:
            lines.append(
                f"\n**Pruning candidates** ({len(candidates)}): "
                + ", ".join(f"`{c}`" for c in candidates)
                + "\n\n"
            )
        else:
            lines.append("\nNo pruning candidates at this threshold.\n\n")

    return lines


# ── Phase B: LOO ablation for marginal families ───────────────────────────────

def run_ablation_phase(conn) -> list[str]:
    print("\n── Phase B: Marginal-family ablation ────────────────────────────────")

    # Reference metrics from the saved canonical model (no retraining needed)
    full_meta = json.loads(MODEL_META_PATH.read_text())
    ref = full_meta["metrics"]["test"]
    ref_auc, ref_brier = ref["auc"], ref["brier"]
    n_ref = len(full_meta["feature_cols"])
    print(
        f"  Reference (full model, {n_ref} features): "
        f"AUC={ref_auc:.4f}, Brier={ref_brier:.4f}"
    )

    params_path = ROOT / "models" / "aoe4_predict" / "lgbm_best_params.json"
    best_params = json.loads(params_path.read_text()) if params_path.exists() else {}

    rows = []

    for label, families in ABLATION_CONFIGS.items():
        print(f"\n  [{label}] families = {sorted(families)}")

        df = extend_training_features(conn, None, families)

        # Hard check: all expected features present and non-all-NaN
        _check_df_for_families(df, families, label)

        # extend_training_features pulls raw SQL columns from player_stats_ext for ALL
        # families regardless of which are enabled (only Python-derived features are
        # gated). Explicitly drop every column belonging to a disabled family so the
        # model sees a clean ablation.
        dropped_families = ALL_ACTIVE - families
        cols_to_drop = []
        for fam in dropped_families:
            for feat in FAMILY_FEATURES[fam]:
                if feat in df.columns:
                    cols_to_drop.append(feat)
        if cols_to_drop:
            print(f"  Dropping {len(cols_to_drop)} raw columns from disabled families")
            df = df.drop(columns=cols_to_drop)

        # Train to a temp path so the canonical model is never overwritten
        with (
            tempfile.NamedTemporaryFile(suffix=".txt",  delete=False) as mf,
            tempfile.NamedTemporaryFile(suffix=".json", delete=False) as jf,
        ):
            tmp_model, tmp_meta = Path(mf.name), Path(jf.name)

        _, meta = train(
            df,
            model_path=tmp_model,
            meta_path=tmp_meta,
            params=dict(best_params),  # copy — train() pops n_estimators
        )

        tmp_model.unlink(missing_ok=True)
        tmp_meta.unlink(missing_ok=True)

        # Hard checks on trained model
        _check_model_has_families(meta, families, label)
        _check_metrics_valid(meta, label)

        m = meta["metrics"]["test"]
        rows.append({
            "label": label,
            "n_features": len(meta["feature_cols"]),
            "auc":   m["auc"],
            "brier": m["brier"],
            "dauc":  round(m["auc"]   - ref_auc,   4),
            "dbrier": round(m["brier"] - ref_brier, 4),
        })

    # Console table
    hdr = f"  {'Config':<22}  {'Feats':>6}  {'AUC':>7}  {'Brier':>7}  {'ΔAUC':>8}  {'ΔBrier':>8}"
    sep = "  " + "-" * (len(hdr) - 2)
    print(f"\n{hdr}\n{sep}")
    print(f"  {'full_model (ref)':<22}  {n_ref:>6}  {ref_auc:>7.4f}  {ref_brier:>7.4f}  {'0.0000':>8}  {'0.0000':>8}")
    for r in rows:
        print(
            f"  {r['label']:<22}  {r['n_features']:>6}  {r['auc']:>7.4f}  "
            f"{r['brier']:>7.4f}  {r['dauc']:>+8.4f}  {r['dbrier']:>+8.4f}"
        )

    # Markdown
    lines: list[str] = []
    lines.append("## Marginal-Family Ablation\n\n")
    lines.append(
        f"Reference: full model ({n_ref} features), "
        f"AUC={ref_auc:.4f}, Brier={ref_brier:.4f}. "
        f"Families P1/P3/P4 significance already established in prior reports.\n\n"
    )
    lines.append("| Config | Features | AUC | Brier | ΔAUC | ΔBrier |\n")
    lines.append("|---|---|---|---|---|---|\n")
    lines.append(
        f"| full_model | {n_ref} | {ref_auc:.4f} | {ref_brier:.4f} | — | — |\n"
    )
    for r in rows:
        s = lambda x: f"+{x:.4f}" if x >= 0 else f"{x:.4f}"
        lines.append(
            f"| {r['label']} | {r['n_features']} | {r['auc']:.4f} | "
            f"{r['brier']:.4f} | {s(r['dauc'])} | {s(r['dbrier'])} |\n"
        )
    lines.append("\n")

    return lines


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    conn = get_conn()

    n_rows = conn.execute("SELECT count(*) FROM training_features").fetchone()[0]
    if n_rows < 10_000:
        raise RuntimeError(
            f"training_features has only {n_rows:,} rows — "
            "run `python -m aoe4_predict train --seasons 10,11` first"
        )
    print(f"training_features: {n_rows:,} rows")

    report: list[str] = ["# Feature Pruning Report\n\n"]

    shap_lines = run_shap_phase(conn)
    ablation_lines = run_ablation_phase(conn)

    report.extend(ablation_lines)
    report.extend(shap_lines)

    conn.close()

    out = ROOT / "reports" / "feature_pruning_report.md"
    out.write_text("".join(report))
    print(f"\nReport saved → {out}")


if __name__ == "__main__":
    main()
