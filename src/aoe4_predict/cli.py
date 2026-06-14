"""
CLI entry point.

Commands:
  ingest   -- Load JSON.gz files into DuckDB
  quality  -- Print data quality report
  train    -- Build features and train LightGBM
  evaluate -- Load saved model and evaluate on test set
  predict  -- Predict match outcome for two player IDs
"""
import argparse
import sys
from pathlib import Path


def _cmd_ingest(args) -> None:
    from .ingest import ingest_all
    data_dir = Path(args.data_dir) if args.data_dir else None
    db_path = Path(args.db) if args.db else None
    seasons = [int(s) for s in args.seasons.split(",")] if args.seasons else None
    ingest_all(data_dir=data_dir, db_path=db_path, seasons=seasons, skip_existing=not args.force)


def _cmd_ingest_metadata(args) -> None:
    from .db import get_conn, ingest_metadata
    db_path = Path(args.db) if args.db else None
    conn = get_conn(db_path)
    ingest_metadata(conn, Path(args.metadata_dir))
    conn.close()


def _cmd_quality(args) -> None:
    from .data_quality import run_quality_report
    from .db import get_conn
    db_path = Path(args.db) if args.db else None
    conn = get_conn(db_path, read_only=True)
    run_quality_report(conn=conn, save=True)
    conn.close()


def _parse_families(args) -> set[str]:
    """Return the set of extra feature families to activate from CLI flags."""
    from .features_extra import FAMILY_FEATURES
    families = set()
    flag_map = {
        "add_civ_recency":       "civ_recency",
        "add_mmr_trend":         "mmr_trend",
        "add_adjusted_form":     "adjusted_form",
        "add_duration_profile":  "duration_profile",
        "add_head_to_head":      "head_to_head",
        "add_map_archetypes":    "map_archetypes",
        "add_patch_priors":      "patch_priors",
        "add_low_history_detail":"low_history_detail",
        "add_activity_session":  "activity_session",
        "add_time_server":       "time_server",
        "add_elo":               "elo",
    }
    for attr, name in flag_map.items():
        if getattr(args, attr, False):
            families.add(name)
    if getattr(args, "add_all_families", False):
        from .features_extra import DISABLED_FAMILIES
        families = set(FAMILY_FEATURES.keys()) - DISABLED_FAMILIES
    return families


def _cmd_train(args) -> None:
    import gc
    import time
    from .config import DEFAULT_TRAIN_SEASONS
    from .db import get_conn
    from .features import build_civ_matchup_priors, build_player_stats, build_training_features
    from .features_extra import extend_training_features
    from .model import train as train_model

    db_path = Path(args.db) if args.db else None
    seasons = [int(s) for s in args.seasons.split(",")] if args.seasons else DEFAULT_TRAIN_SEASONS
    test_seasons = [int(s) for s in args.test_seasons.split(",")] if getattr(args, "test_seasons", None) else None
    model_path = Path(args.model) if args.model else None
    families = _parse_families(args)

    # When a test holdout season is given, load train+test seasons together so
    # test rows are present in training_features for evaluation.
    all_seasons = sorted(set(seasons) | set(test_seasons)) if test_seasons else seasons

    # Auto-name model file from season tags when --model is not specified
    if model_path is None and test_seasons:
        train_tag  = "s" + "s".join(str(s) for s in seasons)
        test_tag   = "s" + "s".join(str(s) for s in test_seasons)
        model_path = Path(f"models/aoe4_predict/lgbm_{train_tag}_test_{test_tag}.txt")
    meta_path = (
        model_path.parent / (model_path.stem + "_meta.json")
        if model_path else None
    )

    print(f"Training on seasons: {seasons}" + (f"  (test holdout: {test_seasons})" if test_seasons else ""))
    if families:
        print(f"Extra feature families: {sorted(families)}")
    conn = get_conn(db_path)

    t0 = time.time()
    print("\n1. Building player_stats...")
    build_player_stats(conn)

    print("\n2. Building civ matchup priors...")
    build_civ_matchup_priors(conn)

    print("\n3. Building training features...")
    df = build_training_features(conn, train_seasons=all_seasons)

    if families:
        print("\n3b. Adding extended feature families...")
        # Release Python df and close + reopen the DuckDB connection to flush the buffer
        # pool (which fills up from building player_stats + training_features). The fresh
        # connection starts with ~0 buffer usage, leaving 8 GB free for the wide join.
        del df
        gc.collect()
        conn.close()
        conn = get_conn(db_path)
        df = extend_training_features(conn, None, families)

    conn.close()

    print(f"\n4. Training model ({len(df):,} rows, {len([c for c in df.columns if c != 'target'])} features)...")
    _, meta = train_model(
        df,
        model_path=model_path,
        meta_path=meta_path,
        test_seasons=test_seasons,
        symmetric_slots=args.symmetric_slots,
    )

    print(f"\nDone in {time.time()-t0:.0f}s")
    print("\n── LightGBM Metrics ──")
    for split, m in meta["metrics"].items():
        print(f"  {split:<6}  AUC={m['auc']:.4f}  LogLoss={m['log_loss']:.4f}  Brier={m['brier']:.4f}")

    if getattr(args, "also_train_xgb", False):
        from .model import train_xgb
        print(f"\n4b. Training XGBoost ({len(df):,} rows)...")
        _, xmeta = train_xgb(df)
        print("\n── XGBoost Metrics ──")
        for split, m in xmeta["metrics"].items():
            print(f"  {split:<6}  AUC={m['auc']:.4f}  LogLoss={m['log_loss']:.4f}  Brier={m['brier']:.4f}")


def _cmd_evaluate(args) -> None:
    from .config import DEFAULT_TRAIN_SEASONS
    from .db import get_conn, table_exists
    from .evaluate import compare_baselines
    from .features import _add_derived_features, build_training_features
    from .features_extra import extend_training_features, FAMILY_FEATURES
    from .model import _predict, _temporal_split, load_model

    db_path = Path(args.db) if args.db else None
    seasons = [int(s) for s in args.seasons.split(",")] if args.seasons else DEFAULT_TRAIN_SEASONS

    # Load model first so we know which features it was trained on
    model, meta = load_model()
    trained_features = set(meta["feature_cols"])

    # Auto-detect which extra families the model needs
    families = _parse_families(args)
    if not families:
        # Auto-detect from model meta: any family whose features overlap trained_features
        for fname, feats in FAMILY_FEATURES.items():
            if any(f in trained_features for f in feats):
                families.add(fname)
        if families:
            print(f"  Auto-detected extra families from saved model: {sorted(families)}")

    conn = get_conn(db_path)
    if table_exists(conn, "training_features"):
        print("  Reading existing training_features table...")
        import pandas as pd
        df = conn.execute("SELECT * FROM training_features").df()
        df = _add_derived_features(df)
    else:
        df = build_training_features(conn, train_seasons=seasons)

    if families:
        df = extend_training_features(conn, df, families)

    conn.close()

    train_df, valid_df, test_df = _temporal_split(df)
    test_preds = _predict(model, test_df, meta["feature_cols"])

    compare_baselines(train_df, test_df, test_preds)


def _cmd_analyze_civ(args) -> None:
    from .config import DEFAULT_TRAIN_SEASONS
    from .db import get_conn
    from .features import _add_derived_features, build_civ_matchup_priors, build_player_stats, build_training_features
    from .features_extra import extend_training_features
    from .civ_analysis import run_civ_familiarity_analysis

    db_path = Path(args.db) if args.db else None
    seasons = [int(s) for s in args.seasons.split(",")] if args.seasons else DEFAULT_TRAIN_SEASONS
    families = _parse_families(args)

    conn = get_conn(db_path)
    build_player_stats(conn)
    build_civ_matchup_priors(conn)
    df = build_training_features(conn, train_seasons=seasons)
    df = _add_derived_features(df)
    if families:
        df = extend_training_features(conn, df, families)

    run_civ_familiarity_analysis(conn, df)
    conn.close()


def _cmd_tune(args) -> None:
    from .config import DEFAULT_TRAIN_SEASONS
    from .db import get_conn
    from .features import _add_derived_features, build_civ_matchup_priors, build_player_stats, build_training_features
    from .features_extra import extend_training_features
    from .tune import run_tune

    db_path = Path(args.db) if args.db else None
    seasons = [int(s) for s in args.seasons.split(",")] if args.seasons else DEFAULT_TRAIN_SEASONS
    families = _parse_families(args)

    conn = get_conn(db_path)
    build_player_stats(conn)
    build_civ_matchup_priors(conn)
    df = build_training_features(conn, train_seasons=seasons)
    df = _add_derived_features(df)
    if families:
        df = extend_training_features(conn, df, families)
    conn.close()

    run_tune(df, model_type=args.model, n_trials=args.n_trials,
             timeout=args.timeout, retrain=not args.no_retrain)


def _cmd_predict(args) -> None:
    from .output import format_prediction
    from .predict import predict_match

    db_path = Path(args.db) if args.db else None

    result = predict_match(
        player_a_id=int(args.player_a),
        player_b_id=int(args.player_b),
        civ_a=args.civ_a,
        civ_b=args.civ_b,
        map_name=args.map,
        db_path=db_path,
    )
    print(format_prediction(result))


def _add_family_flags(p) -> None:
    """Add --add-* feature family flags to a subparser."""
    p.add_argument("--add-civ-recency",        action="store_true", help="P1: time-windowed civ history")
    p.add_argument("--add-mmr-trend",          action="store_true", help="P2: MMR change, slope, volatility")
    p.add_argument("--add-adjusted-form",      action="store_true", help="P3: recent WR over last N games")
    p.add_argument("--add-duration-profile",   action="store_true", help="P4: short/long game WR by duration")
    p.add_argument("--add-head-to-head",       action="store_true", help="P5: cumulative H2H history")
    p.add_argument("--add-map-archetypes",     action="store_true", help="P6: map archetype features (curated + empirical)")
    p.add_argument("--add-patch-priors",       action="store_true", help="P7: patch-age and empirical map priors")
    p.add_argument("--add-low-history-detail", action="store_true", help="P8: granular cold-start flags")
    p.add_argument("--add-activity-session",   action="store_true", help="P9: time-windowed activity counts")
    p.add_argument("--add-time-server",        action="store_true", help="P10: time/server context (stub)")
    p.add_argument("--add-elo",                action="store_true", help="P11: rolling Elo estimate (stub)")
    p.add_argument("--add-all-families",       action="store_true", help="Enable all implemented families")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m aoe4_predict",
        description="AOE4 RM 1v1 match outcome predictor",
    )
    parser.add_argument("--db", default=None, help="Path to DuckDB file (default: aoe4.duckdb)")

    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Load JSON.gz files into DuckDB")
    p_ingest.add_argument("--data-dir", default=None, help="Directory containing JSON.gz files")
    p_ingest.add_argument("--seasons", default=None, help="Comma-separated season numbers, e.g. '10,11'")
    p_ingest.add_argument("--force", action="store_true", help="Re-ingest even if season already exists")
    p_ingest.set_defaults(func=_cmd_ingest)

    # ingest-metadata
    p_imeta = sub.add_parser("ingest-metadata", help="Load map/patch metadata CSVs into DuckDB")
    p_imeta.add_argument("--metadata-dir", default="metadata", help="Directory with map_metadata.csv and patch_metadata.csv")
    p_imeta.set_defaults(func=_cmd_ingest_metadata)

    # quality
    p_qual = sub.add_parser("quality", help="Print data quality report")
    p_qual.set_defaults(func=_cmd_quality)

    # train
    p_train = sub.add_parser("train", help="Build features and train LightGBM")
    p_train.add_argument("--seasons", default=None, help="Comma-separated training season numbers (default: 10,11,12)")
    p_train.add_argument("--test-seasons", default=None,
                         help="Seasons held out as test set (e.g. '11'). --seasons becomes train+valid only.")
    p_train.add_argument("--model", default=None, help="Output model file path")
    p_train.add_argument("--symmetric-slots", action="store_true", help="Augment training rows with player A/B slot swaps")
    p_train.add_argument("--also-train-xgb", action="store_true", help="Also train XGBoost on same features")
    _add_family_flags(p_train)
    p_train.set_defaults(func=_cmd_train)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate saved model against baselines")
    p_eval.add_argument("--seasons", default=None, help="Seasons used for training features")
    _add_family_flags(p_eval)
    p_eval.set_defaults(func=_cmd_evaluate)

    # report
    p_rep = sub.add_parser("report", help="Generate analysis_report.md (leakage audit, subgroups, SHAP)")
    p_rep.add_argument("--output", default=None, help="Output markdown path (default: analysis_report.md)")
    p_rep.add_argument("--model", default=None, help="Model file to report on (default: models/aoe4_predict/lgbm_s10s11s12.txt)")
    p_rep.add_argument("--meta",  default=None, help="Model meta JSON (default: same stem as --model)")
    p_rep.set_defaults(func=lambda a: __import__("aoe4_predict.report", fromlist=["generate_report"]).generate_report(
        report_path=Path(a.output) if a.output else None,
        model_path=Path(a.model) if a.model else None,
        meta_path=Path(a.meta) if a.meta else None,
    ))

    # analyze-civ
    p_civ = sub.add_parser("analyze-civ", help="Skill-stratified civ familiarity analysis")
    p_civ.add_argument("--seasons", default=None, help="Training seasons (default: 10,11,12)")
    _add_family_flags(p_civ)
    p_civ.set_defaults(func=_cmd_analyze_civ)

    # tune
    p_tune = sub.add_parser("tune", help="Hyperparameter tuning via Optuna TPE")
    p_tune.add_argument("--model", default="lgbm", choices=["lgbm", "xgb"], help="Model to tune (default: lgbm)")
    p_tune.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials (default: 50)")
    p_tune.add_argument("--timeout", type=int, default=None, help="Max tuning wall-time in seconds")
    p_tune.add_argument("--no-retrain", action="store_true", help="Skip final re-train with best params")
    p_tune.add_argument("--seasons", default=None, help="Comma-separated training seasons (default: 10,11,12)")
    _add_family_flags(p_tune)
    p_tune.set_defaults(func=_cmd_tune)

    # predict
    p_pred = sub.add_parser("predict", help="Predict match outcome")
    p_pred.add_argument("--player-a", required=True, type=str, help="Profile ID of player A")
    p_pred.add_argument("--player-b", required=True, type=str, help="Profile ID of player B")
    p_pred.add_argument("--civ-a", default=None, help="Civilization for player A (optional)")
    p_pred.add_argument("--civ-b", default=None, help="Civilization for player B (optional)")
    p_pred.add_argument("--map", default=None, help="Map name (optional)")
    p_pred.set_defaults(func=_cmd_predict)

    ns = parser.parse_args(argv)

    try:
        ns.func(ns)
    except KeyboardInterrupt:
        sys.exit(1)


if __name__ == "__main__":
    main()
