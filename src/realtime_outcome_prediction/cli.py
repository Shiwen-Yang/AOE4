import argparse
import json
from pathlib import Path

from .config import AOE4_WORLD_REPO_DIR, CACHE_DIR, DB_PATH, FEATURE_DIR, PARSED_REPLAY_DIR, RAW_REPLAY_DIR

# Default path to the sibling AOE4_Parsing CLI project
_PARSER_PROJECT_DEFAULT = (
    Path(__file__).resolve().parents[3] / "AOE4_Parsing" / "src" / "AoE4ReplayParser.Cli"
)


def _connect_rw(db_path: Path):
    import duckdb

    return duckdb.connect(str(db_path))


def _cmd_hydrate_outcomes(args) -> None:
    from .outcomes import hydrate_outcomes

    conn = _connect_rw(Path(args.db))
    try:
        counts = hydrate_outcomes(
            conn=conn,
            parsed_dir=Path(args.parsed_dir),
            raw_dir=Path(args.raw_dir),
            parser_project=Path(args.parser_project),
            limit=args.limit,
            sleep_seconds=args.sleep_seconds,
            use_official_fallback=not args.no_official_fallback,
        )
        print(json.dumps(counts, indent=2))
    finally:
        conn.close()


def _cmd_train(args) -> None:
    from .model import train_lgbm

    meta = train_lgbm(
        snapshots_path=Path(args.snapshots),
        output_dir=Path(args.output_dir),
        num_leaves=args.num_leaves,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        min_child_samples=args.min_child_samples,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    print(json.dumps(meta, indent=2))


def _cmd_evaluate(args) -> None:
    from .model import evaluate as evaluate_model

    results = evaluate_model(
        snapshots_path=Path(args.snapshots),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(results, indent=2))


def _cmd_build_dataset(args) -> None:
    from .dataset import build_dataset

    report = build_dataset(
        parsed_dir=Path(args.parsed_dir),
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        cache_dir=Path(args.cache_dir),
        db_path=Path(args.db),
        limit=args.limit,
        include_swapped=args.include_swapped,
        refresh_metadata=not args.no_refresh_metadata,
        aoe4world_repo_dir=Path(args.aoe4world_repo_dir),
        allow_profile_id_slot_fallback=args.allow_profile_id_slot_fallback,
        include_delta=args.include_delta,
        include_age_features=args.include_age_features,
        include_fractions=args.include_fractions,
        include_cancel_eff=args.include_cancel_eff,
    )
    print(json.dumps(report, indent=2, default=str))


def _cmd_inspect(args) -> None:
    from .features import read_events
    from .replays import discover_replays, sample_across_id_range

    replays = sample_across_id_range(discover_replays(Path(args.parsed_dir)), args.limit)
    rows = []
    for replay in replays:
        events = read_events(replay.path)
        max_time = max((float(event.get("time_s") or 0.0) for event in events), default=0.0)
        rows.append(
            {
                "replay_id": replay.replay_id,
                "events": len(events),
                "duration_s": round(max_time, 3),
                "slots": sorted({event.get("player_slot") for event in events}),
            }
        )
    print(json.dumps(rows, indent=2))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m realtime_outcome_prediction",
        description="Build realtime AOE4 replay snapshot features",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_hydrate = sub.add_parser(
        "hydrate-outcomes",
        help="Fetch game metadata + outcomes from AoE4World/official APIs",
    )
    p_hydrate.add_argument("--parsed-dir", default=str(PARSED_REPLAY_DIR))
    p_hydrate.add_argument("--raw-dir", default=str(RAW_REPLAY_DIR))
    p_hydrate.add_argument("--db", default=str(DB_PATH))
    p_hydrate.add_argument(
        "--parser-project",
        default=str(_PARSER_PROJECT_DEFAULT),
        help="Path to AoE4ReplayParser.Cli project directory",
    )
    p_hydrate.add_argument("--limit", type=int, default=500)
    p_hydrate.add_argument("--sleep-seconds", type=float, default=1.0)
    p_hydrate.add_argument(
        "--no-official-fallback",
        action="store_true",
        help="Disable fallback to official World's Edge API",
    )
    p_hydrate.set_defaults(func=_cmd_hydrate_outcomes)

    _default_snapshots = str(FEATURE_DIR / "snapshots.parquet")
    _default_model_dir = str(FEATURE_DIR)

    p_train = sub.add_parser("train", help="Train LightGBM on snapshot features")
    p_train.add_argument("--snapshots", default=_default_snapshots)
    p_train.add_argument("--output-dir", default=_default_model_dir)
    p_train.add_argument("--num-leaves", type=int, default=63)
    p_train.add_argument("--n-estimators", type=int, default=500)
    p_train.add_argument("--learning-rate", type=float, default=0.05)
    p_train.add_argument("--min-child-samples", type=int, default=20)
    p_train.add_argument("--early-stopping-rounds", type=int, default=50)
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate saved model on snapshot splits")
    p_eval.add_argument("--snapshots", default=_default_snapshots)
    p_eval.add_argument("--output-dir", default=_default_model_dir)
    p_eval.set_defaults(func=_cmd_evaluate)

    p_build = sub.add_parser("build-dataset", help="Build engineered snapshot features")
    p_build.add_argument("--parsed-dir", default=str(PARSED_REPLAY_DIR))
    p_build.add_argument("--raw-dir", default=str(RAW_REPLAY_DIR))
    p_build.add_argument("--output-dir", default=str(FEATURE_DIR))
    p_build.add_argument("--cache-dir", default=str(CACHE_DIR))
    p_build.add_argument("--aoe4world-repo-dir", default=str(AOE4_WORLD_REPO_DIR))
    p_build.add_argument("--db", default=str(DB_PATH))
    p_build.add_argument("--limit", type=int, default=3_000)
    p_build.add_argument(
        "--include-swapped",
        choices=["none", "train_only", "all"],
        default="train_only",
        help="Generate slot-swapped rows. Default augments train rows only.",
    )
    p_build.add_argument(
        "--no-refresh-metadata",
        action="store_true",
        help="Use the existing AoE4 World git checkout/cache instead of pulling before build.",
    )
    p_build.add_argument(
        "--allow-profile-id-slot-fallback",
        action="store_true",
        help="If no slot mapping is found, assign slot order by profile_id. Disabled by default.",
    )
    p_build.add_argument("--include-delta", action="store_true", help="Add inter-snapshot delta features (F1)")
    p_build.add_argument("--include-age-features", action="store_true", help="Add age-progression timing features (F2)")
    p_build.add_argument("--include-fractions", action="store_true", help="Add resource composition fraction features (F3)")
    p_build.add_argument("--include-cancel-eff", action="store_true", help="Add cancellation efficiency features (F4)")
    p_build.set_defaults(func=_cmd_build_dataset)

    p_inspect = sub.add_parser("inspect", help="Inspect selected parsed replay timelines")
    p_inspect.add_argument("--parsed-dir", default=str(PARSED_REPLAY_DIR))
    p_inspect.add_argument("--limit", type=int, default=5)
    p_inspect.set_defaults(func=_cmd_inspect)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
