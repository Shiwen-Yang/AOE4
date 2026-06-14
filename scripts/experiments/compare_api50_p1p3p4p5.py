"""Compare full-history vs API-capped P1+P3+P4+P5 outcome models.

This experiment trains:
  1. A baseline model with the full P1+P3+P4+P5 families.
  2. A capped-history variant where P1/P4/P5 are recomputed with only each
     player's last 50 pre-match games visible, mirroring one AoE4World recent-
     games page per player.

The "AoE4World full-history subset" is reported as a control row that reuses the
baseline metrics, because with full history it is equivalent to the baseline
feature set for these families.
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
    P1_FEATURES,
    P3_FEATURES,
    P4_FEATURES,
    P5_FEATURES,
    apply_api_cap_p1_p3_p4_p5_overrides,
    build_api_cap_p1_p3_p4_p5_overrides,
    extend_training_features,
)
from aoe4_predict.model import train as train_model

MODEL_DIR = ROOT / "models" / "aoe4_predict"
REPORT_DIR = ROOT / "reports" / "generated"
FAMILIES = {"civ_recency", "adjusted_form", "duration_profile", "head_to_head"}
EXPECTED_EXTRA = set(P1_FEATURES + P3_FEATURES + P4_FEATURES + P5_FEATURES)


def _build_extended_matrix(db_path: Path | None, seasons: list[int]) -> pd.DataFrame:
    conn = get_conn(db_path)
    try:
        print("\n1. Building player_stats...", flush=True)
        build_player_stats(conn)
        print("\n2. Building civ matchup priors...", flush=True)
        build_civ_matchup_priors(conn)
        print("\n3. Building training features...", flush=True)
        df = build_training_features(conn, train_seasons=seasons)
        del df
        gc.collect()
    finally:
        conn.close()

    conn = get_conn(db_path)
    try:
        print("\n4. Adding P1+P3+P4+P5 families...", flush=True)
        df = extend_training_features(conn, None, FAMILIES)
        return df
    finally:
        conn.close()


def _write_summary(
    output_path: Path,
    baseline_meta: dict,
    api_cap_meta: dict,
    baseline_model_path: Path,
    api_cap_model_path: Path,
    visible_match_cap: int,
) -> None:
    rows = [
        {
            "Variant": "Baseline: full P1+P3+P4+P5",
            "Artifact": baseline_model_path.name,
            "Feature count": len(baseline_meta["feature_cols"]),
            "AUC": baseline_meta["metrics"]["test"]["auc"],
            "Log loss": baseline_meta["metrics"]["test"]["log_loss"],
            "Brier": baseline_meta["metrics"]["test"]["brier"],
            "Acc@0.5": baseline_meta["metrics"]["test"]["acc@0.5"],
            "Notes": "DuckDB full-history training semantics.",
        },
        {
            "Variant": "AoE4World full-history subset (identical to baseline)",
            "Artifact": baseline_model_path.name,
            "Feature count": len(baseline_meta["feature_cols"]),
            "AUC": baseline_meta["metrics"]["test"]["auc"],
            "Log loss": baseline_meta["metrics"]["test"]["log_loss"],
            "Brier": baseline_meta["metrics"]["test"]["brier"],
            "Acc@0.5": baseline_meta["metrics"]["test"]["acc@0.5"],
            "Notes": "Control row: same 75 extra features are live-recoverable with full API history.",
        },
        {
            "Variant": f"AoE4World 1-call subset ({visible_match_cap} matches/player)",
            "Artifact": api_cap_model_path.name,
            "Feature count": len(api_cap_meta["feature_cols"]),
            "AUC": api_cap_meta["metrics"]["test"]["auc"],
            "Log loss": api_cap_meta["metrics"]["test"]["log_loss"],
            "Brier": api_cap_meta["metrics"]["test"]["brier"],
            "Acc@0.5": api_cap_meta["metrics"]["test"]["acc@0.5"],
            "Notes": f"P1/P4/P5 recomputed from each player's last {visible_match_cap} visible prior games; P3 unchanged.",
        },
    ]
    df = pd.DataFrame(rows)

    headers = list(df.columns)
    md_rows = [headers] + df.astype(str).values.tolist()
    widths = [
        max(len(str(row[idx])) for row in md_rows)
        for idx in range(len(headers))
    ]

    def render_row(row: list[str]) -> str:
        return "| " + " | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)) + " |"

    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    markdown_table = "\n".join(
        [render_row(md_rows[0]), separator] + [render_row(row) for row in md_rows[1:]]
    )

    lines = [
        "# AoE4World API50 Comparison",
        "",
        "Compares the full-history P1+P3+P4+P5 baseline against an API-capped",
        f"variant that only sees each player's last {visible_match_cap} pre-match games.",
        "",
        markdown_table,
        "",
        "Recoverability notes:",
        f"- `P1 civ_recency`: Model B matches Model A because full history reproduces the same civ counts, win rates, fractions, and `days_since_civ`. Model C degrades once civ usage older than the last {visible_match_cap} visible games matters.",
        "- `P3 adjusted_form`: Model B matches Model A, and Model C should usually match too, because the features only require the most recent 5/10/20 pre-match games.",
        f"- `P4 duration_profile`: Model B matches Model A because full-history lifetime and civ-conditioned duration stats are recoverable. Model C degrades on lifetime-style averages, short/long counts, shares, and derived diffs once players have more than {visible_match_cap} prior games.",
        f"- `P5 head_to_head`: Model B matches Model A because all prior meetings remain visible with full history. Model C only counts prior meetings whose game IDs are visible in both players' capped {visible_match_cap}-game histories.",
        "",
        "Assumptions:",
        "- The full-history AoE4World subset is equivalent to the baseline for these families, so Model B reuses Model A metrics and artifacts.",
        "- The API50 variant isolates degradation from the P1/P3/P4/P5 cap; base features stay unchanged.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"\nSummary written → {output_path}")


def _train_cpu(
    df: pd.DataFrame,
    model_path: Path,
    meta_path: Path,
    symmetric_slots: bool,
    randomize_train_slots: bool,
) -> dict:
    _, meta = train_model(
        df,
        model_path=model_path,
        meta_path=meta_path,
        params={},
        symmetric_slots=symmetric_slots,
        randomize_train_slots_flag=randomize_train_slots,
    )
    return meta


def _validate_comparison_contract(
    baseline_cols: list[str],
    api_cap_cols: list[str],
    baseline_feature_cols: list[str],
    baseline_meta: dict,
    api_cap_meta: dict,
) -> None:
    if baseline_cols != api_cap_cols:
        raise ValueError("API50 matrix schema drifted from the baseline matrix.")
    if baseline_feature_cols != api_cap_meta["feature_cols"]:
        raise ValueError("Baseline and API50 models trained with different feature columns.")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare full-history vs API-capped P1+P3+P4+P5 models.")
    parser.add_argument("--db", default=None, help="Path to DuckDB file (default: aoe4.duckdb)")
    parser.add_argument("--seasons", default="10,11,12", help="Comma-separated training seasons")
    parser.add_argument(
        "--symmetric-slots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Augment the train split with slot-swapped rows (default: disabled for memory safety)",
    )
    parser.add_argument(
        "--randomize-train-slots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Swap A/B on roughly half the train rows without increasing row count (default: enabled)",
    )
    parser.add_argument(
        "--output-prefix",
        default="lgbm_s10s11s12_p1p3p4p5",
        help="Artifact stem prefix under models/aoe4_predict/",
    )
    parser.add_argument(
        "--report-path",
        default=str(REPORT_DIR / "aoe4world_api50_p1p3p4p5_comparison.md"),
        help="Markdown summary output path",
    )
    parser.add_argument(
        "--api-cap",
        type=int,
        default=50,
        help="Visible prior-game cap for the API-style variant (default: 50)",
    )
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else None
    seasons = [int(s) for s in args.seasons.split(",") if s]
    report_path = Path(args.report_path)
    baseline_model_path = MODEL_DIR / f"{args.output_prefix}_full.txt"
    baseline_meta_path = baseline_model_path.parent / f"{baseline_model_path.stem}_meta.json"
    api_variant_tag = f"api{args.api_cap}"
    api50_model_path = MODEL_DIR / f"{args.output_prefix}_{api_variant_tag}.txt"
    api50_meta_path = api50_model_path.parent / f"{api50_model_path.stem}_meta.json"

    t0 = time.time()
    print(f"Training seasons: {seasons}")
    print(f"Extra feature families: {sorted(FAMILIES)}")
    print(f"Symmetric slots: {args.symmetric_slots}")
    print(f"Randomize train slots: {args.randomize_train_slots}")
    print(f"API cap: {args.api_cap}")
    print("Requested device: cpu")

    baseline_ready = baseline_model_path.exists() and baseline_meta_path.exists()
    api50_ready = api50_model_path.exists() and api50_meta_path.exists()
    baseline_cols: list[str] | None = None

    if baseline_ready:
        print("\n5. Reusing existing baseline artifact...", flush=True)
        baseline_meta = _load_json(baseline_meta_path)
        baseline_feature_cols = baseline_meta["feature_cols"]
        print(f"Baseline artifact reused → {baseline_model_path}", flush=True)
    else:
        df = _build_extended_matrix(db_path, seasons)
        baseline_extra = EXPECTED_EXTRA.intersection(df.columns)
        missing = EXPECTED_EXTRA - baseline_extra
        if missing:
            raise ValueError(f"Baseline matrix is missing expected extended features: {sorted(missing)}")
        baseline_cols = list(df.columns)

        print(f"\n5. Training baseline ({len(df):,} rows)...", flush=True)
        baseline_meta = _train_cpu(
            df,
            model_path=baseline_model_path,
            meta_path=baseline_meta_path,
            symmetric_slots=args.symmetric_slots,
            randomize_train_slots=args.randomize_train_slots,
        )
        baseline_feature_cols = baseline_meta["feature_cols"]
        print("Baseline trained on cpu.", flush=True)
        del df
        gc.collect()

    if api50_ready:
        print("\n6. Reusing existing API50 artifact...", flush=True)
        api50_meta = _load_json(api50_meta_path)
        api50_cols = baseline_cols or baseline_feature_cols
        print(f"API50 artifact reused → {api50_model_path}", flush=True)
    else:
        print("\n6. Building matrix for API50 variant...", flush=True)
        df = _build_extended_matrix(db_path, seasons)
        api50_cols = list(df.columns)
        if baseline_cols is None:
            baseline_cols = api50_cols

        conn = get_conn(db_path)
        try:
            print("\n7. Building API50 overrides...", flush=True)
            overrides = build_api_cap_p1_p3_p4_p5_overrides(conn, visible_match_cap=args.api_cap)
        finally:
            conn.close()

        print("\n8. Applying API50 overrides...", flush=True)
        df = apply_api_cap_p1_p3_p4_p5_overrides(df, overrides)
        del overrides
        gc.collect()

        print(f"\n9. Training API50 variant ({len(df):,} rows)...", flush=True)
        api50_meta = _train_cpu(
            df,
            model_path=api50_model_path,
            meta_path=api50_meta_path,
            symmetric_slots=args.symmetric_slots,
            randomize_train_slots=args.randomize_train_slots,
        )
        print("API50 variant trained on cpu.", flush=True)

        _validate_comparison_contract(baseline_cols, api50_cols, baseline_feature_cols, baseline_meta, api50_meta)

    _write_summary(report_path, baseline_meta, api50_meta, baseline_model_path, api50_model_path, args.api_cap)
    print(f"\nDone in {time.time() - t0:.0f}s")
    print(json.dumps({
        "baseline_model": str(baseline_model_path),
        "api50_model": str(api50_model_path),
        "report_path": str(report_path),
    }, indent=2))


if __name__ == "__main__":
    main()
