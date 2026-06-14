"""
CLI for the non_1v1_prediction (team 4v4) investigation.

    PYTHONPATH=src python -m non_1v1_prediction download --mode rm_4v4 --seasons 9,10,11
    PYTHONPATH=src python -m non_1v1_prediction ingest   --mode rm_4v4 --seasons 9,10,11
    PYTHONPATH=src python -m non_1v1_prediction train     --mode rm_4v4
    PYTHONPATH=src python -m non_1v1_prediction evaluate   --mode rm_4v4
    PYTHONPATH=src python -m non_1v1_prediction report     --mode rm_4v4
"""
import argparse

from .config import DEFAULT_MODE, TEAM_SEASONS


def _seasons(arg: str | None) -> list[int]:
    if not arg:
        return TEAM_SEASONS
    return [int(s) for s in arg.split(",") if s.strip()]


def _cmd_download(args):
    from .download import download_dumps
    download_dumps(mode=args.mode, seasons=_seasons(args.seasons), force=args.force)


def _cmd_ingest(args):
    from .ingest import ingest_all
    ingest_all(mode=args.mode, seasons=_seasons(args.seasons),
               db_path=args.db, skip_existing=not args.force)


def _cmd_train(args):
    from . import model as M
    from .features import build_dataset

    df = build_dataset(args.mode, _seasons(args.seasons), db_path=args.db)
    print(f"  {len(df):,} matches")
    train, valid, test = M.temporal_split(df)
    model = M.train_lgbm(M.augment_with_team_swaps(train), M.augment_with_team_swaps(valid))
    p = M.predict(model, test)
    metrics = M.compute_metrics(test["target"].to_numpy(), p)
    print("  test metrics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
    M.save_model(model, args.mode, _seasons(args.seasons), metrics)


def _cmd_evaluate(args):
    from . import baselines as bl
    from . import model as M
    from .features import build_dataset

    df = build_dataset(args.mode, _seasons(args.seasons), db_path=args.db, rebuild=False)
    train, valid, test = M.temporal_split(df)
    train_aug = M.augment_with_team_swaps(train)
    model = M.train_lgbm(train_aug, M.augment_with_team_swaps(valid))
    y = test["target"].to_numpy()
    print("  Constant  :", M.compute_metrics(y, bl.ConstantBaseline().fit(train_aug).predict_proba(test)))
    print("  MMR-logit :", M.compute_metrics(y, bl.MMRMeanDiffLogistic().fit(train_aug).predict_proba(test)))
    print("  LightGBM  :", M.compute_metrics(y, M.predict(model, test)))


def _cmd_build_network(args):
    from . import network as N
    from .config import TEAM_MODES, TEAMMATE_X
    from .db import get_conn

    summ = N.build_all(modes=TEAM_MODES, db_path=args.db)
    print("Network summary:", summ)

    conn = get_conn(args.db)
    try:
        print(f"\nx-validation (same-team vs random opposite-team), TEAMMATE_X={TEAMMATE_X}:")
        print(N.threshold_distribution(conn).to_string(index=False))
        out = N.write_network_report(conn, modes=TEAM_MODES)
        print(f"\nNetwork report written to {out}")
    finally:
        conn.close()


def _cmd_report(args):
    from .report import generate_report
    generate_report(mode=args.mode, seasons=_seasons(args.seasons), db_path=args.db,
                    rebuild=not args.no_rebuild)


def main():
    parser = argparse.ArgumentParser(prog="python -m non_1v1_prediction")
    parser.add_argument("--db", default=None, help="override aoe4_team.duckdb path")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--mode", default=DEFAULT_MODE)
        p.add_argument("--seasons", default=None, help="comma list, e.g. 9,10,11")

    p = sub.add_parser("download"); add_common(p); p.add_argument("--force", action="store_true")
    p.set_defaults(func=_cmd_download)
    p = sub.add_parser("ingest"); add_common(p); p.add_argument("--force", action="store_true")
    p.set_defaults(func=_cmd_ingest)
    p = sub.add_parser("build-network"); add_common(p); p.set_defaults(func=_cmd_build_network)
    p = sub.add_parser("train"); add_common(p); p.set_defaults(func=_cmd_train)
    p = sub.add_parser("evaluate"); add_common(p); p.set_defaults(func=_cmd_evaluate)
    p = sub.add_parser("report"); add_common(p)
    p.add_argument("--no-rebuild", action="store_true", help="reuse existing feature tables")
    p.set_defaults(func=_cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
