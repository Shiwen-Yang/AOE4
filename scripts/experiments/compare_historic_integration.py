"""Does full-history career signal help on top of a recent-window model?

Trains three LightGBM outcome models on the same temporal split and compares them:

  1. full_history  — the existing uncapped P1+P3+P4+P5 model (reference).
  2. recent_only   — an honest "one aoe4world page" mock: every history-derived
                     feature (base lifetime/season/civ/map counts AND the P1/P4/P5
                     extended families) reflects only each player's last N prior
                     games. MMR/rating are retained (the API returns them per game).
  3. dual          — recent_only PLUS a full-history career block added as separate
                     columns (peak/avg MMR, career WR, civ proficiency, form-vs-
                     career), standing in for a refreshed career-summary cache.

The headline is the dual − recent_only delta: positive ⇒ historic signal is
complementary to recent form when both are available as distinct features.
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
    apply_career_block,
    apply_recent_only_base_overrides,
    build_api_cap_p1_p3_p4_p5_overrides,
    build_career_block,
    build_recent_only_base_overrides,
    extend_training_features,
)
from aoe4_predict.model import train as train_model

MODEL_DIR = ROOT / "models" / "aoe4_predict"
REPORT_DIR = ROOT / "reports" / "generated"
FAMILIES = {"civ_recency", "adjusted_form", "duration_profile", "head_to_head"}

VARIANTS = ("full_history", "recent_only", "dual")
VARIANT_NOTES = {
    "full_history": "Uncapped P1+P3+P4+P5 — full DuckDB history.",
    "recent_only": "Base + P1/P4/P5 capped to last {cap} prior games; P3 unchanged. MMR/rating kept.",
    "dual": "recent_only + full-history career block (peak/avg MMR, career WR, civ WR, form-vs-career).",
}


def _build_extended_matrix(db_path: Path | None, seasons: list[int]) -> pd.DataFrame:
    """Materialize player_stats / priors / training_features, return P1+P3+P4+P5 matrix."""
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
    """Build the feature matrix for one variant. Rebuilds the base matrix per call
    (memory-safe — the harness never holds two full matrices at once)."""
    df = _build_extended_matrix(db_path, seasons)
    if kind == "full_history":
        return df

    conn = get_conn(db_path)
    try:
        print(f"  Building recent-window overrides (cap={cap})...", flush=True)
        api_over = build_api_cap_p1_p3_p4_p5_overrides(conn, visible_match_cap=cap)
        base_over = build_recent_only_base_overrides(conn, visible_match_cap=cap)
        career = build_career_block(conn) if kind == "dual" else None
    finally:
        conn.close()

    df = apply_api_cap_p1_p3_p4_p5_overrides(df, api_over)
    df = apply_recent_only_base_overrides(df, base_over)
    del api_over, base_over
    gc.collect()
    if kind == "dual":
        df = apply_career_block(df, career)
        del career
        gc.collect()
    return df


def _train_variant(
    db_path: Path | None,
    seasons: list[int],
    cap: int,
    kind: str,
    model_path: Path,
    meta_path: Path,
    symmetric_slots: bool,
    randomize_train_slots: bool,
) -> dict:
    if model_path.exists() and meta_path.exists():
        print(f"\n[{kind}] reusing existing artifact → {model_path.name}", flush=True)
        return json.loads(meta_path.read_text())

    print(f"\n[{kind}] building matrix...", flush=True)
    df = _variant_df(db_path, seasons, cap, kind)
    print(f"[{kind}] training ({len(df):,} rows, {len(df.columns)} cols)...", flush=True)
    _, meta = train_model(
        df,
        model_path=model_path,
        meta_path=meta_path,
        params={},
        symmetric_slots=symmetric_slots,
        randomize_train_slots_flag=randomize_train_slots,
    )
    del df
    gc.collect()
    return meta


def _write_summary(output_path: Path, metas: dict[str, dict], paths: dict[str, Path], cap: int) -> None:
    rows = []
    for kind in VARIANTS:
        m = metas[kind]["metrics"]["test"]
        rows.append({
            "Variant": kind,
            "Artifact": paths[kind].name,
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

    def render(row: list[str]) -> str:
        return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) + " |"

    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    table = "\n".join([render(md_rows[0]), sep] + [render(r) for r in md_rows[1:]])

    ro = metas["recent_only"]["metrics"]["test"]
    du = metas["dual"]["metrics"]["test"]
    delta = {
        "AUC": round(du["auc"] - ro["auc"], 4),
        "Log loss": round(du["log_loss"] - ro["log_loss"], 4),
        "Brier": round(du["brier"] - ro["brier"], 4),
        "Acc@0.5": round(du["acc@0.5"] - ro["acc@0.5"], 4),
    }

    lines = [
        "# Historic Integration — Dual-Horizon Experiment",
        "",
        f"Recent-window cap N = {cap}. Training/eval on the standard temporal split.",
        "",
        table,
        "",
        "## Headline: dual − recent_only",
        "",
        f"- AUC:      {delta['AUC']:+.4f}  (positive = historic helps)",
        f"- Log loss: {delta['Log loss']:+.4f}  (negative = better)",
        f"- Brier:    {delta['Brier']:+.4f}  (negative = better)",
        f"- Acc@0.5:  {delta['Acc@0.5']:+.4f}  (positive = better)",
        "",
        "Interpretation: `recent_only` is the honest aoe4world-page mock (everything",
        "history-derived capped to the last N games, MMR/rating retained). `dual` adds",
        "full-history career summaries as separate columns. A positive AUC / negative",
        "Brier delta is evidence that a periodically-refreshed career-summary cache",
        "would lift a live recent-window model — motivating Phases 2-4 (exp-decay blend,",
        "form-vs-baseline deltas, career/experience scalars).",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"\nSummary written → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=None, help="Path to DuckDB file (default: aoe4.duckdb)")
    parser.add_argument("--seasons", default="10,11,12", help="Comma-separated training seasons")
    parser.add_argument("--cap", type=int, default=30, help="Recent-window visibility cap (default: 30)")
    parser.add_argument(
        "--symmetric-slots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Augment train split with slot-swapped rows (default: disabled for memory safety)",
    )
    parser.add_argument(
        "--randomize-train-slots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Swap A/B on ~half the train rows without growing row count (default: enabled)",
    )
    parser.add_argument(
        "--output-prefix",
        default="lgbm_historic_integration",
        help="Artifact stem prefix under models/aoe4_predict/",
    )
    parser.add_argument(
        "--report-path",
        default=str(REPORT_DIR / "historic_integration_dual_horizon.md"),
        help="Markdown summary output path",
    )
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else None
    seasons = [int(s) for s in args.seasons.split(",") if s]
    report_path = Path(args.report_path)

    t0 = time.time()
    print(f"Training seasons: {seasons}")
    print(f"Recent-window cap: {args.cap}")
    print(f"Symmetric slots: {args.symmetric_slots} | Randomize slots: {args.randomize_train_slots}")

    metas: dict[str, dict] = {}
    paths: dict[str, Path] = {}
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
    print(json.dumps({kind: str(paths[kind]) for kind in VARIANTS}, indent=2))


if __name__ == "__main__":
    main()
