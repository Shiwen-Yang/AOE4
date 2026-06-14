"""
L1 Logistic Ablation — 5 iterative fit→eval→change cycles.

Features: base training_features + day-level (7d/30d) rolling stats.
No per-game extended window functions (P1-P9 families are skipped).
Train/val/test are all within S9+S10 (70/15/15 temporal split).
S11 is never loaded.

Run:
    python scripts/experiments/run_l1_ablation.py [--db path/to/aoe4.duckdb]

Output:
    reports/generated/l1_ablation_s9s10.md
"""
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

TRAIN_SEASONS = [9, 10]
REPORT_PATH   = Path("reports/generated/l1_ablation_s9s10.md")
PRIOR_GAMES   = 5  # smoothing prior for day-level win rates


# ── data loading ──────────────────────────────────────────────────────────────

def _compute_day_features(conn):
    """Day-level rolling stats (7d / 30d) for every player-day. ~9s on all seasons."""
    return conn.execute("""
        WITH player_days AS (
            SELECT
                p.profile_id,
                CAST(g.started_at AS DATE)                AS game_day,
                COUNT(*)                                   AS n_games,
                SUM(CASE WHEN p.result THEN 1 ELSE 0 END) AS n_wins
            FROM participants p
            JOIN games g USING (game_id)
            GROUP BY p.profile_id, CAST(g.started_at AS DATE)
        )
        SELECT
            profile_id,
            game_day,
            COALESCE(SUM(n_games) OVER (PARTITION BY profile_id ORDER BY game_day
                RANGE BETWEEN INTERVAL 7 DAYS PRECEDING AND INTERVAL 1 DAYS PRECEDING), 0) AS games_7d,
            COALESCE(SUM(n_wins)  OVER (PARTITION BY profile_id ORDER BY game_day
                RANGE BETWEEN INTERVAL 7 DAYS PRECEDING AND INTERVAL 1 DAYS PRECEDING), 0) AS wins_7d,
            COALESCE(SUM(n_games) OVER (PARTITION BY profile_id ORDER BY game_day
                RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND INTERVAL 1 DAYS PRECEDING), 0) AS games_30d,
            COALESCE(SUM(n_wins)  OVER (PARTITION BY profile_id ORDER BY game_day
                RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND INTERVAL 1 DAYS PRECEDING), 0) AS wins_30d
        FROM player_days
    """).df()


def load_data(db_path=None):
    from aoe4_predict.db import get_conn
    from aoe4_predict.features import _add_derived_features

    conn = get_conn(db_path)

    seasons_sql = ",".join(str(s) for s in TRAIN_SEASONS)
    print(f"Loading base training_features for S{TRAIN_SEASONS}...", flush=True)
    df = conn.execute(
        f"SELECT * FROM training_features WHERE season IN ({seasons_sql})"
    ).df()
    print(f"  {len(df):,} base rows × {len(df.columns)} cols", flush=True)

    df = _add_derived_features(df)

    print("Computing day-level rolling features (7d / 30d across all history)...", flush=True)
    t0 = time.time()
    day_feats = _compute_day_features(conn)
    conn.close()
    print(f"  {len(day_feats):,} player-day rows in {time.time() - t0:.1f}s", flush=True)

    # Normalise types for the join: both sides must be datetime64 at midnight
    day_feats["game_day"] = pd.to_datetime(day_feats["game_day"])
    df["_game_day"] = df["started_at"].dt.normalize()

    for side in ("a", "b"):
        pid_col = f"profile_id_{side}"
        day_feats_side = day_feats.rename(columns={
            "profile_id": pid_col,
            "game_day":   "_game_day",
            "games_7d":   f"games_7d_{side}",
            "wins_7d":    f"wins_7d_{side}",
            "games_30d":  f"games_30d_{side}",
            "wins_30d":   f"wins_30d_{side}",
        })
        df = df.merge(day_feats_side, on=[pid_col, "_game_day"], how="left")

    df = df.drop(columns=["_game_day"])

    # Smoothed day-level win rates and diffs
    for w in (7, 30):
        for s in ("a", "b"):
            g = df[f"games_{w}d_{s}"].fillna(0)
            w_ = df[f"wins_{w}d_{s}"].fillna(0)
            df[f"wr_{w}d_{s}"] = (w_ + PRIOR_GAMES * 0.5) / (g + PRIOR_GAMES)
        df[f"wr_{w}d_diff"] = df[f"wr_{w}d_a"] - df[f"wr_{w}d_b"]

    print(f"  Final dataset: {len(df):,} rows × {len(df.columns)} cols", flush=True)
    return df


def split_data(df):
    from aoe4_predict.model import _temporal_split
    train_df, valid_df, test_df = _temporal_split(df)

    def date_range(d):
        ts = d["started_at"]
        return f"{ts.min().date()} → {ts.max().date()}"

    print("\nTemporal 70/15/15 split within S9+S10:", flush=True)
    print(f"  Train : {len(train_df):>8,}  {date_range(train_df)}", flush=True)
    print(f"  Valid : {len(valid_df):>8,}  {date_range(valid_df)}", flush=True)
    print(f"  Test  : {len(test_df):>8,}  {date_range(test_df)}", flush=True)
    return train_df, valid_df, test_df


# ── cycle runner ──────────────────────────────────────────────────────────────

def run_cycle(label, model, train_df, valid_df, test_df, prior_model=None, cycle_num=0):
    from aoe4_predict.evaluate import evaluate

    print(f"\n{'─' * 62}", flush=True)
    print(f"Cycle {cycle_num}: {label}", flush=True)
    print(f"{'─' * 62}", flush=True)

    t0 = time.time()
    model.fit(train_df, valid_df)
    elapsed = time.time() - t0

    val_preds  = model.predict_proba(valid_df)
    test_preds = model.predict_proba(test_df)
    val_m  = evaluate(valid_df["target"].values, val_preds)
    test_m = evaluate(test_df["target"].values,  test_preds)

    bd      = model.feature_breakdown()
    total   = sum(bd.values())
    nonzero = len(model.selected_features())

    print(f"  Val  AUC={val_m['auc']:.4f}  Brier={val_m['brier']:.4f}  LogLoss={val_m['log_loss']:.4f}", flush=True)
    print(f"  Test AUC={test_m['auc']:.4f}  Brier={test_m['brier']:.4f}  LogLoss={test_m['log_loss']:.4f}", flush=True)
    print(f"  Features: {total} total  ({bd['numeric']} num / {bd['ohe']} ohe / "
          f"{bd['compound_ohe']} compound / {bd['poly']} poly / {bd['interaction']} ix)")
    print(f"  L1 selected: {nonzero}/{total}  ({total - nonzero} zeroed)  [{elapsed:.0f}s]", flush=True)

    ct   = model.coef_table()
    top20 = ct.head(20)
    print(f"\n  Top 20 by |coef|:", flush=True)
    for _, row in top20.iterrows():
        sign = "+" if row["coef"] > 0 else "-"
        print(f"    {sign}{row['abs_coef']:.4f}  {row['feature']}", flush=True)

    if prior_model is not None:
        prior_sel = set(prior_model.selected_features())
        cur_sel   = set(model.selected_features())
        newly_zero   = sorted(prior_sel - cur_sel)[:8]
        newly_active = sorted(cur_sel - prior_sel)
        if newly_zero:
            print(f"\n  Newly zeroed vs prior cycle ({len(prior_sel - cur_sel)} total): {newly_zero}{'...' if len(prior_sel - cur_sel) > 8 else ''}", flush=True)
        if newly_active:
            print(f"  Newly selected vs prior cycle ({len(newly_active)} total): {sorted(newly_active)[:8]}{'...' if len(newly_active) > 8 else ''}", flush=True)

    return {
        "label":     label,
        "cycle":     cycle_num,
        "model":     model,
        "val_m":     val_m,
        "test_m":    test_m,
        "breakdown": bd,
        "nonzero":   nonzero,
        "total":     total,
        "elapsed":   elapsed,
    }


# ── markdown report ───────────────────────────────────────────────────────────

def generate_report(results, train_df, valid_df, test_df):
    def date_range(d):
        ts = d["started_at"]
        return f"{ts.min().date()} – {ts.max().date()}"

    lines = [
        "# L1 Logistic Ablation: S9+S10 (70/15/15 temporal split)",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Features: base `training_features` + day-level 7d/30d rolling stats. "
        "No P1-P9 extended families. S11 never loaded.",
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
        m     = r["test_m"]
        delta = f"+{m['auc'] - auc_c1:.4f}" if r["cycle"] > 1 else "—"
        lines.append(
            f"| C{r['cycle']} | {r['label']} | {r['total']} | {r['model'].best_C} | "
            f"{r['val_m']['brier']:.4f} | {m['auc']:.4f} | {m['brier']:.4f} | "
            f"{m['log_loss']:.4f} | {r['nonzero']} | {delta} |"
        )

    lines += ["", "---", ""]

    for r in results:
        m     = r["test_m"]
        model = r["model"]
        bd    = r["breakdown"]
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
            f"Best C: `{model.best_C}` — {total} features "
            f"({bd['numeric']} numeric / {bd['ohe']} OHE / {bd['compound_ohe']} compound / "
            f"{bd['poly']} poly / {bd['interaction']} interaction) — "
            f"{r['nonzero']} non-zero ({total - r['nonzero']} zeroed by L1)",
            "",
            "### Top 20 Coefficients by |coef|",
            "",
            "| Feature | Coef |",
            "|---|---|",
        ]
        for _, row in model.coef_table().head(20).iterrows():
            lines.append(f"| `{row['feature']}` | {row['coef']:+.4f} |")

        if model.include_ohe or model.compound_ohe_pairs:
            ohe_sum = model.ohe_summary(top_n=5)
            if ohe_sum:
                lines += ["", "### OHE Column Summary (top 5 per group by |coef|)", ""]
                for group, entries in ohe_sum.items():
                    lines.append(f"**{group}**")
                    for feat, coef in entries:
                        lines.append(f"- `{feat}`: {coef:+.4f}")
                    lines.append("")

        lines += ["", "---", ""]

    # Cycle 5 deep-dive
    c5 = next((r for r in results if r["cycle"] == 5), None)
    if c5 is not None:
        model = c5["model"]
        ct_nz = model.coef_table()
        ct_nz = ct_nz[ct_nz["coef"] != 0]
        lines += ["## Cycle 5 Deep Dive", ""]

        for prefix, title in [
            ("civ_a_x_civ_b=", "Civ Matchup Pairs (civ_a × civ_b)"),
            ("season_x_civ_a=", "Season × Civ A"),
            ("season_x_civ_b=", "Season × Civ B"),
        ]:
            rows = ct_nz[ct_nz["feature"].str.startswith(prefix)].head(15)
            if not rows.empty:
                lines += [f"### {title}", "", "| Feature | Coef |", "|---|---|"]
                for _, row in rows.iterrows():
                    lines.append(f"| `{row['feature']}` | {row['coef']:+.4f} |")
                lines.append("")

        poly_rows = ct_nz[ct_nz["feature"].str.startswith("poly_")].head(15)
        if not poly_rows.empty:
            lines += ["### Polynomial Terms Selected", "", "| Feature | Coef |", "|---|---|"]
            for _, row in poly_rows.iterrows():
                lines.append(f"| `{row['feature']}` | {row['coef']:+.4f} |")
            lines.append("")

        lines.append("---")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print(f"\nReport written to {REPORT_PATH}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    db_path = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--db" and i + 2 < len(sys.argv):
            db_path = sys.argv[i + 2]

    from aoe4_predict.baselines import EnhancedL1LogisticBaseline

    t_total = time.time()

    df = load_data(db_path)
    train_df, valid_df, test_df = split_data(df)

    results = []

    # Extra interaction specs using the new day-level win-rate features
    DAY_INTERACTIONS = [
        ("ix_skill_x_wr7d_diff",  "skill_diff", "wr_7d_diff"),
        ("ix_skill_x_wr30d_diff", "skill_diff", "wr_30d_diff"),
        ("ix_wr7d_a_x_wr30d_a",   "wr_7d_a",   "wr_30d_a"),
        ("ix_wr7d_b_x_wr30d_b",   "wr_7d_b",   "wr_30d_b"),
    ]
    BASE_INTERACTIONS = EnhancedL1LogisticBaseline._DEFAULT_INTERACTIONS + DAY_INTERACTIONS

    # ── Cycle 1: Numeric baseline ─────────────────────────────────────────────
    r1 = run_cycle(
        "Numeric only",
        EnhancedL1LogisticBaseline(include_ohe=False, include_interactions=False),
        train_df, valid_df, test_df, cycle_num=1,
    )
    results.append(r1)

    # ── Cycle 2: + OHE primary categoricals ──────────────────────────────────
    r2 = run_cycle(
        "+ OHE (civ/map/patch/season)",
        EnhancedL1LogisticBaseline(
            include_ohe=True,
            ohe_cols=["civ_a", "civ_b", "map", "patch", "season"],
            include_interactions=False,
        ),
        train_df, valid_df, test_df, prior_model=r1["model"], cycle_num=2,
    )
    results.append(r2)

    # ── Cycle 3: + all interaction features ──────────────────────────────────
    r3 = run_cycle(
        "+ OHE + interactions (base + day-level)",
        EnhancedL1LogisticBaseline(
            include_ohe=True,
            ohe_cols=["civ_a", "civ_b", "map", "patch", "season"],
            include_interactions=True,
            interaction_specs=BASE_INTERACTIONS,
        ),
        train_df, valid_df, test_df, prior_model=r2["model"], cycle_num=3,
    )
    results.append(r3)

    # ── Cycle 4: Drop zeroed numerics; add civ×map compound OHE ──────────────
    c3_zeroed_num  = [f for f in r3["model"].zeroed_features() if f in r3["model"]._num_cols_fitted]
    c3_selected_ix = [
        spec for spec in BASE_INTERACTIONS
        if spec[0] in r3["model"].selected_features()
    ]

    r4 = run_cycle(
        "+ civ×map compound OHE (trimmed numerics)",
        EnhancedL1LogisticBaseline(
            include_ohe=True,
            ohe_cols=["civ_a", "civ_b", "map", "patch", "season"],
            compound_ohe_pairs=[("civ_a", "map"), ("civ_b", "map")],
            include_interactions=True,
            interaction_specs=c3_selected_ix or BASE_INTERACTIONS,
            drop_features=c3_zeroed_num,
            c_grid=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        ),
        train_df, valid_df, test_df, prior_model=r3["model"], cycle_num=4,
    )
    results.append(r4)

    # ── Cycle 5: + civ×civ matchup OHE, season×civ, poly on top-8 ───────────
    c4_model = r4["model"]
    c4_zeroed_num = [f for f in c4_model.zeroed_features() if f in c4_model._num_cols_fitted]
    c4_selected_ix = [
        spec for spec in (c3_selected_ix or BASE_INTERACTIONS)
        if spec[0] in c4_model.selected_features()
    ]

    # OHE groups that had at least one non-zero entry in C4
    c4_sel_set   = set(c4_model.selected_features())
    c4_base_ohe  = ["civ_a", "civ_b", "map", "patch", "season"]
    c4_ohe_kept  = [col for col in c4_base_ohe if any(f.startswith(col + "_") for f in c4_sel_set)] or c4_base_ohe

    # Top-8 numeric features from C4 for polynomial expansion
    top8_numeric = [
        row["feature"] for _, row in c4_model.coef_table().iterrows()
        if row["feature"] in c4_model._num_cols_fitted
    ][:8]

    r5 = run_cycle(
        "+ civ×civ matchup + season×civ + poly top-8",
        EnhancedL1LogisticBaseline(
            include_ohe=True,
            ohe_cols=c4_ohe_kept,
            compound_ohe_pairs=[
                ("civ_a", "civ_b"),
                ("civ_a", "map"),
                ("civ_b", "map"),
                ("season", "civ_a"),
                ("season", "civ_b"),
            ],
            poly_features_on=top8_numeric,
            include_interactions=True,
            interaction_specs=c4_selected_ix or BASE_INTERACTIONS,
            drop_features=c4_zeroed_num,
            c_grid=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        ),
        train_df, valid_df, test_df, prior_model=c4_model, cycle_num=5,
    )
    results.append(r5)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'=' * 80}", flush=True)
    auc_c1 = results[0]["test_m"]["auc"]
    hdr = f"{'Cycle':<8} {'Variant':<45} {'Feats':>6} {'C':>8} {'ValBr':>7} {'TestAUC':>8} {'TestBr':>7} {'ΔAUC':>7}"
    print(hdr, flush=True)
    print("─" * len(hdr), flush=True)
    for r in results:
        m     = r["test_m"]
        delta = f"+{m['auc'] - auc_c1:.4f}" if r["cycle"] > 1 else "—"
        print(
            f"C{r['cycle']:<7} {r['label']:<45} {r['total']:>6} "
            f"{str(r['model'].best_C):>8} {r['val_m']['brier']:>7.4f} "
            f"{m['auc']:>8.4f} {m['brier']:>7.4f} {delta:>7}"
        )

    generate_report(results, train_df, valid_df, test_df)
    print(f"\nAll done in {(time.time() - t_total) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
