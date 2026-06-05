from __future__ import annotations

import argparse
from pathlib import Path

from aoe4_predict.db import get_conn

from .candidates import label_balanced_candidates, summarize_labels
from .config import DB_PATH, SAMPLE_GROUP_BALANCED, SAMPLE_GROUP_TOP100
from .downloader import download_group
from .parser import parse_downloaded
from .reports import write_report
from .schema import init_schema
from .top_players import label_top100_games


def _conn(args, read_only: bool = False):
    return get_conn(Path(args.db) if args.db else DB_PATH, read_only=read_only)


def _cmd_init_schema(args) -> None:
    conn = _conn(args)
    init_schema(conn)
    conn.close()
    print("Replay harvest schema initialized.")


def _cmd_label_balanced(args) -> None:
    conn = _conn(args)
    init_schema(conn)
    counts = label_balanced_candidates(
        conn,
        limit=args.limit,
        season=args.season,
        patch=args.patch,
        sample_group=args.group,
    )
    conn.close()
    print(counts)


def _cmd_label_top100(args) -> None:
    conn = _conn(args)
    init_schema(conn)
    counts = label_top100_games(conn)
    conn.close()
    print(counts)


def _cmd_candidates(args) -> None:
    conn = _conn(args, read_only=True)
    rows = summarize_labels(conn)
    conn.close()
    for row in rows:
        print(row)


def _cmd_download(args) -> None:
    conn = _conn(args)
    init_schema(conn)
    counts = download_group(
        conn,
        sample_group=args.group,
        limit=args.limit,
        sleep_min=args.sleep_min,
        sleep_max=args.sleep_max,
    )
    conn.close()
    print(counts)


def _cmd_report(args) -> None:
    conn = _conn(args, read_only=True)
    report = write_report(conn)
    conn.close()
    print(report)


def _cmd_parse_downloaded(args) -> None:
    conn = _conn(args)
    init_schema(conn)
    counts = parse_downloaded(
        conn,
        limit=args.limit,
        parser_version=args.parser_version,
        sample_group=args.group,
        catalog_dir=Path(args.catalog_dir) if args.catalog_dir else None,
        raw=args.raw,
    )
    conn.close()
    print(counts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m replay_harvest",
        description="Harvest Age of Empires IV replay files for model training.",
    )
    parser.add_argument("--db", default=None, help="DuckDB path, default: /home/shiwen/GitHub/AOE4/aoe4.duckdb")

    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-schema", help="Create replay harvest tables")
    p_init.set_defaults(func=_cmd_init_schema)

    p_bal = sub.add_parser("label-balanced", help="Label balanced RM 1v1 replay candidates")
    p_bal.add_argument("--limit", type=int, default=10_000)
    p_bal.add_argument("--season", type=int, default=None)
    p_bal.add_argument("--patch", default=None)
    p_bal.add_argument("--group", default=SAMPLE_GROUP_BALANCED)
    p_bal.set_defaults(func=_cmd_label_balanced)

    p_top = sub.add_parser("label-top100", help="Label all games for top 100 canonical players")
    p_top.set_defaults(func=_cmd_label_top100)

    p_cand = sub.add_parser("candidates", help="Print candidate label summary")
    p_cand.set_defaults(func=_cmd_candidates)

    p_dl = sub.add_parser("download", help="Download labeled replay candidates")
    p_dl.add_argument("--group", default=SAMPLE_GROUP_BALANCED, choices=[SAMPLE_GROUP_BALANCED, SAMPLE_GROUP_TOP100])
    p_dl.add_argument("--limit", type=int, default=1000)
    p_dl.add_argument("--sleep-min", type=float, default=15.0)
    p_dl.add_argument("--sleep-max", type=float, default=30.0)
    p_dl.set_defaults(func=_cmd_download)

    p_parse = sub.add_parser("parse-downloaded", help="Parse downloaded replay files and record parser status")
    p_parse.add_argument("--group", default=None, choices=[SAMPLE_GROUP_BALANCED, SAMPLE_GROUP_TOP100])
    p_parse.add_argument("--limit", type=int, default=100)
    p_parse.add_argument("--parser-version", default="aoe4_parser_cli")
    p_parse.add_argument("--catalog-dir", default=None)
    p_parse.add_argument("--raw", action="store_true")
    p_parse.set_defaults(func=_cmd_parse_downloaded)

    p_report = sub.add_parser("report", help="Write replay sample reports")
    p_report.set_defaults(func=_cmd_report)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
