"""Does a window-temporal block help the standalone API30 model?

Both variants use ONLY each player's last N games (the honest aoe4world recent-
games-page mock — no DB career cache):

  1. recent_only         — base + P1/P4/P5 capped to the last N prior games; P3
                           unchanged. MMR/rating retained.
  2. recent_only + window_temporal — adds calendar-time descriptors of the last-N
                           window (span, games/day, mean & max idle gap, games in
                           last 7/30 days), all recoverable from the page timestamps.

Headline = (recent_only + window_temporal) − recent_only.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aoe4_predict.db import get_conn
from aoe4_predict.features import build_civ_matchup_priors, build_player_stats, build_training_features
from aoe4_predict.features_extra import (
    apply_api_cap_p1_p3_p4_p5_overrides,
    apply_recent_only_base_overrides,
    apply_window_temporal_overrides,
    build_api_cap_p1_p3_p4_p5_overrides,
    build_recent_only_base_overrides,
    build_window_temporal_overrides,
    extend_training_features,
)
from aoe4_predict.model import train as train_model

MODEL_DIR = ROOT / "models" / "aoe4_predict"
REPORT_DIR = ROOT / "reports" / "generated"
FAMILIES = {"civ_recency", "adjusted_form", "duration_profile", "head_to_head"}

VARIANTS = ("recent_only", "window_temporal")
VARIANT_NOTES = {
    "recent_only": "Base + P1/P4/P5 capped to last {cap} prior games; P3 unchanged. MMR/rating kept.",
    "window_temporal": "recent_only + window-temporal block (span, games/day, gaps, 7d/30d activity).",
}


def _build_extended_matrix(db_path: Path | None, seasons: list[int]) -> pd.DataFrame:
    conn = get_conn(db_path)
    try:
        print("\n  Building player_stats...", flush=True)
        build_player_stats(conn)
        print("  Building civ matchup priors...", flush=True)
        build_civ_matchup_priors(conn)
        print("  Building training features...", flush=True)
        df = build_training_features(conn, train_seasons=seasons)
        del df
        gc.collect()
    finally:
        conn.close()

    conn = get_conn(db_path)
    try:
        print("  Adding P1+P3+P4+P5 families...", flush=True)
        return extend_training_features(conn, None, FAMILIES)
    finally:
        conn.close()


def _variant_df(db_path: Path | None, seasons: list[int], cap: int, kind: str) -> pd.DataFrame:
    df = _build_extended_matrix(db_path, seasons)
    conn = get_conn(db_path)
    try:
        print(f"  Building recent-window overrides (cap={cap})...", flush=True)
        api_over = build_api_cap_p1_p3_p4_p5_overrides(conn, visible_match_cap=cap)
        base_over = build_recent_only_base_overrides(conn, visible_match_cap=cap)
        wt_over = build_window_temporal_overrides(conn, visible_match_cap=cap) if kind == "window_temporal" else None
    finally:
        conn.close()

    df = apply_api_cap_p1_p3_p4_p5_overrides(df, api_over)
    df = apply_recent_only_base_overrides(df, base_over)
    del api_over, base_over
    gc.collect()
    if kind == "window_temporal":
        df = apply_window_temporal_overrides(df, wt_over)
        del wt_over
        gc.collect()
    return df


def _train_variant(db_path, seasons, cap, kind, model_path, meta_path, symmetric, randomize) -> dict:
    if model_path.exists() and meta_path.exists():
        print(f"\n[{kind}] reusing existing artifact → {model_path.name}", flush=True)
        return json.loads(meta_path.read_text())
    print(f"\n[{kind}] building matrix...", flush=True)
    df = _variant_df(db_path, seasons, cap, kind)
    print(f"[{kind}] training ({len(df):,} rows, {len(df.columns)} cols)...", flush=True)
    _, meta = train_model(
        df, model_path=model_path, meta_path=meta_path, params={},
        symmetric_slots=symmetric, randomize_train_slots_flag=randomize,
    )
    del df
    gc.collect()
    return meta


def _write_summary(output_path: Path, metas: dict, paths: dict, cap: int) -> None:
    rows = []
    for kind in VARIANTS:
        m = metas[kind]["metrics"]["test"]
        rows.append({
            "Variant": kind,
            "Features": len(metas[kind]["feature_cols"]),
            "AUC": round(m["auc"], 4),
            "Log loss": round(m["log_loss"], 4),
            "Brier": round(m["brier"], 4),
            "Acc@0.5": round(m["acc@0.5"], 4),
            "Notes": VARIANT_NOTES[kind].format(cap=cap),
        })
    df = pd.DataFrame(rows)
    headers = list(df.columns)
    md_rows = [headers] + df.astype(str).values.tolist()
    widths = [max(len(str(r[i])) for r in md_rows) for i in range(len(headers))]
    render = lambda row: "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    table = "\n".join([render(md_rows[0]), sep] + [render(r) for r in md_rows[1:]])

    ro = metas["recent_only"]["metrics"]["test"]
    wt = metas["window_temporal"]["metrics"]["test"]
    lines = [
        "# Window-Temporal Block vs Standalone API30",
        "",
        f"Recent-window cap N = {cap}. Both variants use only the last N games per",
        "player (aoe4world-page mock). The window-temporal block is computable from",
        "the page timestamps alone — no database career cache.",
        "",
        table,
        "",
        "## Headline: window_temporal − recent_only",
        "",
        f"- AUC:      {wt['auc'] - ro['auc']:+.4f}  (positive = the block helps)",
        f"- Log loss: {wt['log_loss'] - ro['log_loss']:+.4f}  (negative = better)",
        f"- Brier:    {wt['brier'] - ro['brier']:+.4f}  (negative = better)",
        f"- Acc@0.5:  {wt['acc@0.5'] - ro['acc@0.5']:+.4f}  (positive = better)",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"\nSummary written → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=None)
    parser.add_argument("--seasons", default="10,11,12")
    parser.add_argument("--cap", type=int, default=30)
    parser.add_argument("--symmetric-slots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--randomize-train-slots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-prefix", default="lgbm_window_temporal")
    parser.add_argument("--report-path", default=str(REPORT_DIR / "window_temporal_vs_api30.md"))
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else None
    seasons = [int(s) for s in args.seasons.split(",") if s]
    report_path = Path(args.report_path)

    t0 = time.time()
    print(f"Training seasons: {seasons} | cap: {args.cap}")

    metas, paths = {}, {}
    for kind in VARIANTS:
        model_path = MODEL_DIR / f"{args.output_prefix}_{kind}_cap{args.cap}.txt"
        meta_path = model_path.parent / f"{model_path.stem}_meta.json"
        paths[kind] = model_path
        metas[kind] = _train_variant(
            db_path, seasons, args.cap, kind, model_path, meta_path,
            args.symmetric_slots, args.randomize_train_slots,
        )

    _write_summary(report_path, metas, paths, args.cap)
    print(f"\nDone in {time.time() - t0:.0f}s")
    print(json.dumps({k: str(paths[k]) for k in VARIANTS}, indent=2))


if __name__ == "__main__":
    main()
