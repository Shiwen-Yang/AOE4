"""
L1 Logistic Ablation — 5 iterative fit→eval→change cycles.

Train/val/test are all within S9+S10 (70/15/15 temporal split).
S11 is never loaded or touched.

Run:
    python run_l1_ablation.py [--db path/to/aoe4.duckdb]

Output:
    reports/l1_ablation_s9s10.md
"""
import sys
import time
from datetime import datetime
from pathlib import Path

TRAIN_SEASONS = [9, 10]
REPORT_PATH   = Path("reports/l1_ablation_s9s10.md")


# ── data loading ──────────────────────────────────────────────────────────────

def load_data(db_path=None):
    from aoe4_predict.db import get_conn
    from aoe4_predict.features_extra import FAMILY_FEATURES, DISABLED_FAMILIES, extend_training_features

    conn = get_conn(db_path)

    # Include all families; P6/P7 are opt-in here (unlike the main pipeline).
    # If their metadata tables are absent, extend_training_features skips them.
    families = set(FAMILY_FEATURES.keys())

    print(f"Loading features for seasons {TRAIN_SEASONS} (all families)...")
    df = extend_training_features(conn, None, families)
    conn.close()

    df = df[df["season"].isin(TRAIN_SEASONS)].reset_index(drop=True)
    print(f"  {len(df):,} rows × {len(df.columns)} columns after S9+S10 filter")
    return df


def split_data(df):
    from aoe4_predict.model import _temporal_split
    train_df, valid_df, test_df = _temporal_split(df)

    def date_range(d):
        ts = d["started_at"]
        return f"{ts.min().date()} → {ts.max().date()}"

    print("\nTemporal 70/15/15 split within S9+S10:")
    print(f"  Train : {len(train_df):>8,}  {date_range(train_df)}")
    print(f"  Valid : {len(valid_df):>8,}  {date_range(valid_df)}")
    print(f"  Test  : {len(test_df):>8,}  {date_range(test_df)}")
    return train_df, valid_df, test_df


# ── cycle runner ──────────────────────────────────────────────────────────────

def run_cycle(
    label: str,
    model,
    train_df,
    valid_df,
    test_df,
    prior_model=None,
    cycle_num: int = 0,
) -> dict:
    from aoe4_predict.evaluate import evaluate

    print(f"\n{'─' * 60}")
    print(f"Cycle {cycle_num}: {label}")
    print(f"{'─' * 60}")

    t0 = time.time()
    model.fit(train_df, valid_df)
    elapsed = time.time() - t0

    val_preds  = model.predict_proba(valid_df)
    test_preds = model.predict_proba(test_df)

    val_m  = evaluate(valid_df["target"].values, val_preds)
    test_m = evaluate(test_df["target"].values,  test_preds)

    bd = model.feature_breakdown()
    total = sum(bd.values())
    nonzero = len(model.selected_features())

    print(f"  Val  AUC={val_m['auc']:.4f}  Brier={val_m['brier']:.4f}  LogLoss={val_m['log_loss']:.4f}")
    print(f"  Test AUC={test_m['auc']:.4f}  Brier={test_m['brier']:.4f}  LogLoss={test_m['log_loss']:.4f}")
    print(f"  Features: {total} total  ({bd['numeric']} num / {bd['ohe']} ohe / "
          f"{bd['compound_ohe']} compound / {bd['poly']} poly / {bd['interaction']} ix)")
    print(f"  L1 selected: {nonzero}/{total}  ({total - nonzero} zeroed)  [{elapsed:.0f}s]")

    ct = model.coef_table()
    top20 = ct.head(20)
    print(f"\n  Top 20 by |coef|:")
    for _, row in top20.iterrows():
        sign = "+" if row["coef"] > 0 else "-"
        print(f"    {sign}{row['abs_coef']:.4f}  {row['feature']}")

    if prior_model is not None:
        prior_selected = set(prior_model.selected_features())
        cur_selected   = set(model.selected_features())
        newly_zero   = sorted(prior_selected - cur_selected)[:10]
        newly_active = sorted(cur_selected - prior_selected)[:10]
        if newly_zero:
            print(f"\n  Newly zeroed vs prior cycle: {newly_zero}")
        if newly_active:
            print(f"  Newly selected vs prior cycle: {newly_active[:5]}{'...' if len(newly_active) > 5 else ''}")

    return {
        "label":    label,
        "cycle":    cycle_num,
        "model":    model,
        "val_m":    val_m,
        "test_m":   test_m,
        "breakdown": bd,
        "nonzero":  nonzero,
        "total":    total,
        "elapsed":  elapsed,
    }


# ── markdown report ───────────────────────────────────────────────────────────

def generate_report(results: list[dict], train_df, valid_df, test_df):
    def date_range(d):
        ts = d["started_at"]
        return f"{ts.min().date()} – {ts.max().date()}"

    lines = [
        "# L1 Logistic Ablation: S9+S10 (70/15/15 temporal split)",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## Data Split",
        "",
        "| Split | Rows | Date Range |",
        "|---|---|---|",
        f"| Train | {len(train_df):,} | {date_range(train_df)} |",
        f"| Valid | {len(valid_df):,} | {date_range(valid_df)} |",
        f"| Test  | {len(test_df):,}  | {date_range(test_df)} |",
        "",
        "---",
        "",
        "## Summary Results",
        "",
        "| Cycle | Variant | Features | Best C | Val Brier | Test AUC | Test Brier | Test LogLoss | Nonzero | ΔAUC vs C1 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    auc_c1 = results[0]["test_m"]["auc"]
    for r in results:
        m  = r["test_m"]
        bd = r["breakdown"]
        total = r["total"]
        delta = f"+{m['auc'] - auc_c1:.4f}" if r["cycle"] > 1 else "—"
        lines.append(
            f"| C{r['cycle']} | {r['label']} | {total} | {r['model'].best_C} | "
            f"{r['val_m']['brier']:.4f} | {m['auc']:.4f} | {m['brier']:.4f} | "
            f"{m['log_loss']:.4f} | {r['nonzero']} | {delta} |"
        )

    lines += ["", "---", ""]

    # Per-cycle detail sections
    for r in results:
        m  = r["test_m"]
        model = r["model"]
        bd = r["breakdown"]
        total = r["total"]

        lines += [
            f"## Cycle {r['cycle']}: {r['label']}",
            "",
            "### Metrics",
            "",
            "| Split | AUC | Brier | LogLoss | Acc@0.5 |",
            "|---|---|---|---|---|",
            f"| Val  | {r['val_m']['auc']:.4f} | {r['val_m']['brier']:.4f} | {r['val_m']['log_loss']:.4f} | {r['val_m']['acc@0.5']:.4f} |",
            f"| Test | {m['auc']:.4f} | {m['brier']:.4f} | {m['log_loss']:.4f} | {m['acc@0.5']:.4f} |",
            "",
            f"Best C: `{model.best_C}` — Features: {total} "
            f"({bd['numeric']} numeric / {bd['ohe']} OHE / {bd['compound_ohe']} compound / "
            f"{bd['poly']} poly / {bd['interaction']} interaction) — "
            f"{r['nonzero']} non-zero ({total - r['nonzero']} zeroed by L1)",
            "",
            "### Top 20 Coefficients by |coef|",
            "",
            "| Feature | Coef |",
            "|---|---|",
        ]
        ct = model.coef_table()
        for _, row in ct.head(20).iterrows():
            lines.append(f"| `{row['feature']}` | {row['coef']:+.4f} |")

        # OHE summary for cycles with categoricals
        if model.include_ohe or model.compound_ohe_pairs:
            ohe_sum = model.ohe_summary(top_n=5)
            if ohe_sum:
                lines += ["", "### OHE Column Summary (top 5 per group, by |coef|)", ""]
                for group, entries in ohe_sum.items():
                    lines.append(f"**{group}**")
                    for feat, coef in entries:
                        lines.append(f"- `{feat}`: {coef:+.4f}")
                    lines.append("")

        lines += ["", "---", ""]

    # Cycle 5 special deep-dive sections
    c5 = next((r for r in results if r["cycle"] == 5), None)
    if c5 is not None:
        model = c5["model"]
        ct = model.coef_table()
        ct_nz = ct[ct["coef"] != 0]

        lines += ["## Cycle 5 Deep Dive", ""]

        # Civ × civ matchup pairs
        matchup_mask = ct_nz["feature"].str.startswith("civ_a_x_civ_b=")
        matchup_rows = ct_nz[matchup_mask].head(15)
        if not matchup_rows.empty:
            lines += [
                "### Top Civ Matchup Pairs (civ_a × civ_b compound OHE)",
                "",
                "| Matchup | Coef |",
                "|---|---|",
            ]
            for _, row in matchup_rows.iterrows():
                lines.append(f"| `{row['feature']}` | {row['coef']:+.4f} |")
            lines.append("")

        # Season × civ signals
        s_civ_mask = ct_nz["feature"].str.startswith("season_x_civ_a=") | ct_nz["feature"].str.startswith("season_x_civ_b=")
        s_civ_rows = ct_nz[s_civ_mask].head(15)
        if not s_civ_rows.empty:
            lines += [
                "### Season × Civ Coefficients",
                "",
                "| Feature | Coef |",
                "|---|---|",
            ]
            for _, row in s_civ_rows.iterrows():
                lines.append(f"| `{row['feature']}` | {row['coef']:+.4f} |")
            lines.append("")

        # Polynomial terms selected
        poly_mask = ct_nz["feature"].str.startswith("poly_")
        poly_rows = ct_nz[poly_mask].head(15)
        if not poly_rows.empty:
            lines += [
                "### Polynomial Terms Selected",
                "",
                "| Feature | Coef |",
                "|---|---|",
            ]
            for _, row in poly_rows.iterrows():
                lines.append(f"| `{row['feature']}` | {row['coef']:+.4f} |")
            lines.append("")

        lines.append("---")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print(f"\nReport written to {REPORT_PATH}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    db_path = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--db" and i + 1 < len(sys.argv) - 1:
            db_path = sys.argv[i + 2]

    from aoe4_predict.baselines import EnhancedL1LogisticBaseline

    t_total = time.time()

    df = load_data(db_path)
    train_df, valid_df, test_df = split_data(df)

    results: list[dict] = []

    # ── Cycle 1: Numeric only ──────────────────────────────────────────────────
    c1_model = EnhancedL1LogisticBaseline(
        include_ohe=False,
        include_interactions=False,
    )
    r1 = run_cycle("Numeric only", c1_model, train_df, valid_df, test_df, cycle_num=1)
    results.append(r1)

    # ── Cycle 2: + OHE primary categoricals ────────────────────────────────────
    c2_model = EnhancedL1LogisticBaseline(
        include_ohe=True,
        ohe_cols=["civ_a", "civ_b", "map", "patch", "season"],
        include_interactions=False,
    )
    r2 = run_cycle(
        "+ OHE (civ/map/patch/season)", c2_model, train_df, valid_df, test_df,
        prior_model=c1_model, cycle_num=2,
    )
    results.append(r2)

    # ── Cycle 3: + domain interaction features ──────────────────────────────────
    c3_model = EnhancedL1LogisticBaseline(
        include_ohe=True,
        ohe_cols=["civ_a", "civ_b", "map", "patch", "season"],
        include_interactions=True,   # all 10 default interactions
    )
    r3 = run_cycle(
        "+ OHE + interactions", c3_model, train_df, valid_df, test_df,
        prior_model=c2_model, cycle_num=3,
    )
    results.append(r3)

    # ── Cycle 4: Deeper categoricals ────────────────────────────────────────────
    # Drop numeric features zeroed in Cycle 3 (they carry no linear signal).
    # Keep only interaction specs that were selected in Cycle 3.
    # Add P6 map taxonomy OHE and civ×map compound OHE.
    from aoe4_predict.features_extra import P6_CATEGORICAL_FEATURES

    c3_zeroed_numerics = [f for f in c3_model.zeroed_features() if f in c3_model._num_cols_fitted]
    c3_selected_ixns = [
        spec for spec in EnhancedL1LogisticBaseline._DEFAULT_INTERACTIONS
        if spec[0] in c3_model.selected_features()
    ]

    c4_ohe_cols = ["civ_a", "civ_b", "map", "patch", "season"] + P6_CATEGORICAL_FEATURES
    c4_model = EnhancedL1LogisticBaseline(
        include_ohe=True,
        ohe_cols=c4_ohe_cols,
        compound_ohe_pairs=[("civ_a", "map"), ("civ_b", "map")],
        include_interactions=True,
        interaction_specs=c3_selected_ixns if c3_selected_ixns else None,
        drop_features=c3_zeroed_numerics,
        c_grid=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    )
    r4 = run_cycle(
        "+ map taxonomy OHE + civ×map compound", c4_model, train_df, valid_df, test_df,
        prior_model=c3_model, cycle_num=4,
    )
    results.append(r4)

    # ── Cycle 5: Maximum push ────────────────────────────────────────────────────
    # Add civ×civ matchup compound OHE, season×civ compound OHE,
    # and polynomial terms on top-8 numeric features from Cycle 4.
    # Drop all zeroed features from Cycle 4 (numeric + any OHE groups fully zeroed).
    c4_zeroed = c4_model.zeroed_features()
    c4_zeroed_numerics = [f for f in c4_zeroed if f in c4_model._num_cols_fitted]

    # Keep OHE column groups that had at least one non-zero entry in Cycle 4
    c4_selected_set = set(c4_model.selected_features())
    c4_ohe_kept = [
        col for col in c4_ohe_cols
        if any(f.startswith(col + "_") for f in c4_selected_set)
    ]
    if not c4_ohe_kept:
        c4_ohe_kept = c4_ohe_cols  # safety fallback

    # Top-8 numeric features by |coef| for poly expansion
    c4_ct = c4_model.coef_table()
    top8_numeric = [
        row["feature"] for _, row in c4_ct.iterrows()
        if row["feature"] in c4_model._num_cols_fitted
    ][:8]

    # Interaction specs: selected in Cycle 4 (or all defaults as fallback)
    c4_selected_ixns = [
        spec for spec in (c3_selected_ixns or EnhancedL1LogisticBaseline._DEFAULT_INTERACTIONS)
        if spec[0] in c4_selected_set
    ]

    c5_model = EnhancedL1LogisticBaseline(
        include_ohe=True,
        ohe_cols=c4_ohe_kept,
        compound_ohe_pairs=[
            ("civ_a", "civ_b"),   # specific matchup
            ("civ_a", "map"),
            ("civ_b", "map"),
            ("season", "civ_a"),  # civ strength per season
            ("season", "civ_b"),
        ],
        poly_features_on=top8_numeric,
        include_interactions=True,
        interaction_specs=c4_selected_ixns if c4_selected_ixns else None,
        drop_features=c4_zeroed_numerics,
        c_grid=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    )
    r5 = run_cycle(
        "+ civ×civ matchup OHE + season×civ + poly top-8", c5_model, train_df, valid_df, test_df,
        prior_model=c4_model, cycle_num=5,
    )
    results.append(r5)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    auc_c1 = results[0]["test_m"]["auc"]
    header = f"{'Cycle':<8} {'Variant':<44} {'Feats':>6} {'C':>8} {'ValBr':>7} {'TestAUC':>8} {'TestBr':>7} {'ΔAUC':>7}"
    print(header)
    print("─" * len(header))
    for r in results:
        m  = r["test_m"]
        delta = f"+{m['auc'] - auc_c1:.4f}" if r["cycle"] > 1 else "—"
        print(
            f"C{r['cycle']:<7} {r['label']:<44} {r['total']:>6} "
            f"{str(r['model'].best_C):>8} {r['val_m']['brier']:>7.4f} "
            f"{m['auc']:>8.4f} {m['brier']:>7.4f} {delta:>7}"
        )

    generate_report(results, train_df, valid_df, test_df)
    print(f"\nDone in {(time.time() - t_total) / 60:.1f} min")


if __name__ == "__main__":
    main()
