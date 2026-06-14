"""Cross-season generalization of the API30 model — justify deploying to S13.

Rolls a 2-season training window forward and always tests on the NEXT, unseen
season (out-of-distribution). Feature config = the raw API30 recent_only mock
(base + P1/P4/P5 capped to each player's last 30 prior games; P3 unchanged;
MMR/rating retained).

For each test season te in {9,10,11,12}: train on {te-2, te-1}, test on te.
Reports out-of-distribution AUC/LogLoss/Brier/Acc + calibration (ECE), alongside
the in-distribution `valid` metrics and the generalization gap (valid − test).
The trend across boundaries is the expected S12→S13 penalty.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aoe4_predict.baselines import MMRLogisticBaseline
from aoe4_predict.db import get_conn
from aoe4_predict.evaluate import calibration_table, evaluate
from aoe4_predict.features import build_civ_matchup_priors, build_player_stats, build_training_features
from aoe4_predict.features_extra import (
    apply_api_cap_p1_p3_p4_p5_overrides,
    apply_recent_only_base_overrides,
    build_api_cap_p1_p3_p4_p5_overrides,
    build_recent_only_base_overrides,
    extend_training_features,
)
from aoe4_predict.model import _predict, load_model
from aoe4_predict.model import train as train_model

MODEL_DIR = ROOT / "models" / "aoe4_predict"
REPORT_DIR = ROOT / "reports" / "generated"
FAMILIES = {"civ_recency", "adjusted_form", "duration_profile", "head_to_head"}


def _ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: Σ (nᵢ/N)·|empirical − predicted| over bins."""
    tbl = calibration_table(y_true, y_pred, n_bins=n_bins)
    n = tbl["n"].to_numpy(dtype=float)
    if n.sum() == 0:
        return float("nan")
    return float((n * tbl["gap"].abs().to_numpy()).sum() / n.sum())


def _build_pair_df(db_path: Path | None, train_seasons: list[int], te: int, cap: int) -> pd.DataFrame:
    """API30 recent_only matrix for the 3-season slice [*train_seasons, te]."""
    slice_seasons = [*train_seasons, te]
    conn = get_conn(db_path)
    try:
        print(f"  Building training_features for {slice_seasons}...", flush=True)
        df = build_training_features(conn, train_seasons=slice_seasons)
        del df
        gc.collect()
        df = extend_training_features(conn, None, FAMILIES)
        print(f"  Building recent-window overrides (cap={cap})...", flush=True)
        api_over = build_api_cap_p1_p3_p4_p5_overrides(conn, visible_match_cap=cap)
        base_over = build_recent_only_base_overrides(conn, visible_match_cap=cap)
    finally:
        conn.close()

    df = apply_api_cap_p1_p3_p4_p5_overrides(df, api_over)
    df = apply_recent_only_base_overrides(df, base_over)
    del api_over, base_over
    gc.collect()
    return df


def _run_pair(db_path, te, cap, with_baseline, randomize) -> dict:
    train_seasons = [te - 2, te - 1]
    df = _build_pair_df(db_path, train_seasons, te, cap)

    # Retain the unseen-season slice for calibration / baseline before train() consumes df.
    test_slice = df[df["season"] == te].copy()

    baseline_metrics = None
    if with_baseline:
        train_mask = df["season"].isin(train_seasons)
        bl = MMRLogisticBaseline().fit(df.loc[train_mask])
        baseline_metrics = evaluate(test_slice["target"].values, bl.predict_proba(test_slice))

    model_path = MODEL_DIR / f"lgbm_xseason_recent_only_train{train_seasons[0]}{train_seasons[1]}_test{te}.txt"
    meta_path = model_path.parent / f"{model_path.stem}_meta.json"
    print(f"\n[test S{te} | train S{train_seasons}] training ({len(df):,} rows)...", flush=True)
    _, meta = train_model(
        df, model_path=model_path, meta_path=meta_path, params={},
        test_seasons=[te], randomize_train_slots_flag=randomize,
    )
    del df
    gc.collect()

    # Calibration on the unseen season (reload the saved booster → predict retained slice).
    model, saved_meta = load_model(model_path, meta_path)
    y_true = test_slice["target"].values
    y_pred = _predict(model, test_slice, saved_meta["feature_cols"])
    ece = _ece(y_true, y_pred)
    calib = calibration_table(y_true, y_pred, n_bins=10)
    del test_slice
    gc.collect()

    return {
        "test_season": te,
        "train_seasons": train_seasons,
        "n_test": int(len(y_true)),
        "test": meta["metrics"]["test"],
        "valid": meta["metrics"]["valid"],
        "ece": round(ece, 4),
        "baseline": baseline_metrics,
        "calibration": calib,
        "model": model_path.name,
    }


def _write_report(path: Path, results: list[dict], cap: int, with_baseline: bool) -> None:
    cols = ["Train", "Test", "n_test", "AUC", "LogLoss", "Brier", "Acc@0.5", "ECE",
            "valid AUC", "AUC gap", "Brier gap"]
    if with_baseline:
        cols += ["MMR-base AUC"]
    rows = []
    for r in results:
        t, v = r["test"], r["valid"]
        row = [
            f"S{r['train_seasons'][0]}+S{r['train_seasons'][1]}",
            f"S{r['test_season']}",
            f"{r['n_test']:,}",
            f"{t['auc']:.4f}", f"{t['log_loss']:.4f}", f"{t['brier']:.4f}",
            f"{t['acc@0.5']:.4f}", f"{r['ece']:.4f}",
            f"{v['auc']:.4f}",
            f"{v['auc'] - t['auc']:+.4f}",
            f"{t['brier'] - v['brier']:+.4f}",
        ]
        if with_baseline:
            row.append(f"{r['baseline']['auc']:.4f}" if r['baseline'] else "—")
        rows.append(row)

    md = [cols] + rows
    widths = [max(len(str(r[i])) for r in md) for i in range(len(cols))]
    render = lambda r: "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"
    table = "\n".join([render(md[0]), sep] + [render(r) for r in rows])

    auc_gaps = [r["valid"]["auc"] - r["test"]["auc"] for r in results]
    test_aucs = [r["test"]["auc"] for r in results]
    eces = [r["ece"] for r in results]

    lines = [
        "# Cross-Season Generalization — API30 (recent_only)",
        "",
        f"Recent-window cap N = {cap}. Sliding 2-season train window; each row is tested",
        "on the **next, unseen** season (out-of-distribution). `valid` = in-distribution",
        "tail of the training seasons. `AUC gap` = valid − test (the seasonal penalty);",
        "`Brier gap` = test − valid. `ECE` = expected calibration error on the unseen season.",
        "",
        table,
        "",
        "## Aggregate",
        "",
        f"- Out-of-distribution test AUC: mean {np.mean(test_aucs):.4f}, "
        f"range {min(test_aucs):.4f}–{max(test_aucs):.4f}",
        f"- Mean valid→test AUC gap: {np.mean(auc_gaps):+.4f}",
        f"- Mean ECE on unseen season: {np.mean(eces):.4f}",
        "",
        "## Interpretation",
        "",
        "A small, stable AUC gap and low ECE across boundaries indicate the model transfers",
        "to a future unseen season — justifying applying the S10+S11+S12 model to S13. The",
        "S10+S11→S12 row is the closest proxy for the actual S12→S13 deployment.",
        "",
        "### Reliability — closest proxy (S10+S11 → S12)",
        "",
    ]
    proxy = next((r for r in results if r["test_season"] == 12), results[-1])
    ct = proxy["calibration"]
    lines.append("| predicted_wr | empirical_wr | gap | n |")
    lines.append("| --- | --- | --- | --- |")
    for _, row in ct.iterrows():
        lines.append(f"| {row['predicted_wr']:.3f} | {row['empirical_wr']:.3f} | {row['gap']:+.3f} | {int(row['n']):,} |")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"\nReport written → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=None)
    parser.add_argument("--cap", type=int, default=30)
    parser.add_argument("--test-seasons", default="9,10,11,12",
                        help="Comma-separated seasons to test on (train = the 2 prior seasons each)")
    parser.add_argument("--with-baseline", action="store_true",
                        help="Add an MMR-logistic reference column per test season")
    parser.add_argument("--randomize-train-slots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report-path", default=str(REPORT_DIR / "cross_season_generalization_api30.md"))
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else None
    test_seasons = [int(s) for s in args.test_seasons.split(",") if s]
    report_path = Path(args.report_path)

    t0 = time.time()
    print(f"Test seasons: {test_seasons} (train = 2 prior seasons each) | cap: {args.cap}")

    # player_stats + civ_matchup_priors span every season — build once.
    print("\nBuilding shared player_stats + civ_matchup_priors (once)...", flush=True)
    conn = get_conn(db_path)
    try:
        build_player_stats(conn)
        build_civ_matchup_priors(conn)
    finally:
        conn.close()

    results = []
    for te in test_seasons:
        results.append(_run_pair(db_path, te, args.cap, args.with_baseline, args.randomize_train_slots))

    _write_report(report_path, results, args.cap, args.with_baseline)
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
