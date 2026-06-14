import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    CACHE_DIR,
    DB_PATH,
    DEFAULT_LIMIT,
    DEFAULT_TRAIN_FRAC,
    DEFAULT_VALID_FRAC,
    FEATURE_DIR,
    AOE4_WORLD_REPO_DIR,
    PARSED_REPLAY_DIR,
    RAW_REPLAY_DIR,
)
from .features import build_match_snapshots, feature_manifest, read_events
from .labels import LabelResolver
from .metadata import build_pbgid_index, load_or_update_aoe4world_repo
from .replays import assign_match_splits, discover_replays, sample_across_id_range


def _connect_duckdb(db_path: Path):
    try:
        import duckdb

        return duckdb.connect(str(db_path), read_only=True)
    except Exception:
        return None


def _write_frame(df: pd.DataFrame, path_stem: Path) -> str:
    parquet_path = path_stem.with_suffix(".parquet")
    csv_path = path_stem.with_suffix(".csv")
    try:
        df.to_parquet(parquet_path, index=False)
        return str(parquet_path)
    except Exception:
        df.to_csv(csv_path, index=False)
        return str(csv_path)


def _patch_counts(matches: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter = Counter()
    for row in matches:
        patch = row.get("patch")
        counts[str(patch) if patch is not None else "unknown"] += 1
    return dict(counts)


def _row_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter = Counter()
    for row in rows:
        counts[str(row.get(key, "unknown"))] += 1
    return dict(counts)


def _unknown_metadata_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    totals: Counter = Counter()
    for row in rows:
        for key, value in row.items():
            if key.endswith("_unknown_metadata_count") and isinstance(value, (int, float)):
                totals[key] += float(value)
    return {k: round(v, 4) for k, v in sorted(totals.items())}


def build_dataset(
    parsed_dir: Path = PARSED_REPLAY_DIR,
    raw_dir: Path = RAW_REPLAY_DIR,
    output_dir: Path = FEATURE_DIR,
    cache_dir: Path = CACHE_DIR,
    db_path: Path = DB_PATH,
    limit: int = DEFAULT_LIMIT,
    include_swapped: str = "train_only",
    refresh_metadata: bool = True,
    aoe4world_repo_dir: Path = AOE4_WORLD_REPO_DIR,
    allow_profile_id_slot_fallback: bool = False,
    include_delta: bool = False,
    include_age_features: bool = False,
    include_fractions: bool = False,
    include_cancel_eff: bool = False,
) -> dict[str, Any]:
    """Build snapshot features and write the engineered dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_or_update_aoe4world_repo(
        cache_dir=cache_dir,
        repo_dir=aoe4world_repo_dir,
        update=refresh_metadata,
    )
    pbgid_index = build_pbgid_index(metadata)

    conn = _connect_duckdb(db_path)
    resolver = LabelResolver(
        conn=conn,
        raw_dir=raw_dir,
        parsed_dir=parsed_dir,
        allow_profile_id_fallback=allow_profile_id_slot_fallback,
    )

    # Filter to hydrated game_ids first, then sample, so all three splits
    # are populated regardless of which games have been hydrated so far.
    # Fetch all hydrated IDs from the DB without an IN clause (avoids scanning
    # a huge comma-separated list against the 12M-row participants table).
    all_replays = discover_replays(parsed_dir)
    if conn is not None:
        try:
            jsonl_ids = {r.replay_id for r in all_replays}
            db_hydrated = {
                row[0]
                for row in conn.execute(
                    """
                    SELECT game_id FROM participants
                    WHERE player_slot IS NOT NULL
                    GROUP BY game_id
                    HAVING count(*) = 2
                       AND count(CASE WHEN result IS NOT NULL THEN 1 END) = 2
                    """
                ).fetchall()
            }
            hydrated = jsonl_ids & db_hydrated
            all_replays = [r for r in all_replays if r.replay_id in hydrated]
        except Exception:
            pass

    replays = sample_across_id_range(all_replays, limit)
    splits = assign_match_splits(
        [replay.replay_id for replay in replays],
        train_frac=DEFAULT_TRAIN_FRAC,
        valid_frac=DEFAULT_VALID_FRAC,
    )

    match_rows: list[dict[str, Any]] = []
    skip_counts: Counter = Counter()

    # Stats tracked without keeping all rows in memory
    row_count_by_split: Counter = Counter()
    row_count_by_minute: Counter = Counter()
    row_count_by_phase: Counter = Counter()
    unknown_meta_totals: Counter = Counter()
    snapshot_total = 0
    snapshot_feature_cols: list[str] = []

    # Stream snapshot rows to parquet in batches to avoid OOM on large inputs.
    # Each batch is written to a temp chunk file; they're merged at the end via
    # DuckDB which fills missing columns (pbgid columns vary per game) with NULL.
    parquet_path = output_dir / "snapshots.parquet"
    _chunk_dir = output_dir / "_chunks"
    _chunk_dir.mkdir(exist_ok=True)
    _pending: list[dict[str, Any]] = []
    _chunk_idx = 0
    _FLUSH_EVERY = 3_000  # rows (not games)

    def _flush_pending() -> None:
        nonlocal _chunk_idx, snapshot_feature_cols
        if not _pending:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        batch_df = pd.DataFrame(_pending)
        _pending.clear()
        # Drop per-entity pbgid columns — too sparse to use directly as features
        batch_df = batch_df[[c for c in batch_df.columns if "_pbgid_" not in c]]
        if not snapshot_feature_cols:
            snapshot_feature_cols = list(batch_df.columns)
        chunk_path = _chunk_dir / f"chunk_{_chunk_idx:04d}.parquet"
        pq.write_table(pa.Table.from_pandas(batch_df), str(chunk_path))
        _chunk_idx += 1

    for replay in replays:
        split = splits[replay.replay_id]
        label, skip_reason = resolver.resolve(replay.replay_id)
        if skip_reason:
            skip_counts[skip_reason] += 1
            match_rows.append(
                {
                    "replay_id": replay.replay_id,
                    "split": split,
                    "usable": False,
                    "skip_reason": skip_reason,
                }
            )
            continue

        try:
            events = read_events(replay.path)
        except Exception as exc:
            skip_counts["parse_error"] += 1
            match_rows.append(
                {
                    "replay_id": replay.replay_id,
                    "split": split,
                    "usable": False,
                    "skip_reason": f"parse_error:{type(exc).__name__}",
                }
            )
            continue

        should_swap = include_swapped == "all" or (include_swapped == "train_only" and split == "train")
        rows, stats = build_match_snapshots(
            replay_id=replay.replay_id,
            events=events,
            pbgid_index=pbgid_index,
            target=label.target,
            split=split,
            include_swapped=should_swap,
            include_delta=include_delta,
            include_age_features=include_age_features,
            include_fractions=include_fractions,
            include_cancel_eff=include_cancel_eff,
        )
        if not rows:
            skip_counts["no_snapshots"] += 1
            match_rows.append(
                {
                    "replay_id": replay.replay_id,
                    "split": split,
                    "usable": False,
                    "skip_reason": "no_snapshots",
                    "target": label.target,
                    **label.metadata,
                    **stats,
                }
            )
            continue

        for row in rows:
            row_count_by_split[str(row.get("split", "unknown"))] += 1
            row_count_by_minute[str(int(row.get("snapshot_minute", 0)))] += 1
            row_count_by_phase[str(row.get("snapshot_phase", "unknown"))] += 1
            for k, v in row.items():
                if k.endswith("_unknown_metadata_count") and isinstance(v, (int, float)):
                    unknown_meta_totals[k] += float(v)

        snapshot_total += len(rows)
        _pending.extend(rows)
        if len(_pending) >= _FLUSH_EVERY:
            _flush_pending()

        match_rows.append(
            {
                "replay_id": replay.replay_id,
                "split": split,
                "usable": True,
                "skip_reason": None,
                "target": label.target,
                "slot1_profile_id": label.slot_profile_ids.get(1),
                "slot2_profile_id": label.slot_profile_ids.get(2),
                **label.metadata,
                **stats,
            }
        )

    _flush_pending()

    if conn is not None:
        conn.close()

    # Merge chunks into a single parquet, filling missing columns with NULL
    import duckdb as _ddb
    chunk_glob = str(_chunk_dir / "chunk_*.parquet")
    _ddb.execute(
        f"COPY (SELECT * FROM read_parquet('{chunk_glob}')) TO '{parquet_path}' (FORMAT PARQUET)"
    )
    import shutil
    shutil.rmtree(_chunk_dir, ignore_errors=True)

    matches_df = pd.DataFrame(match_rows)
    snapshots_path = str(parquet_path)
    matches_path = _write_frame(matches_df, output_dir / "matches")

    manifest = feature_manifest(snapshot_feature_cols)
    manifest.update(
        {
            "output_files": {
                "snapshots": snapshots_path,
                "matches": matches_path,
            },
            "augmentation": {
                "include_swapped": include_swapped,
                "valid_test_augmented_by_default": False,
            },
            "aoe4world_metadata": {
                "source": metadata.get("source"),
                "repo_dir": metadata.get("repo_dir"),
                "revision": metadata.get("revision"),
                "fetched_at_readable": metadata.get("fetched_at_readable"),
                "update_attempted": metadata.get("update_attempted"),
                "update_error": metadata.get("update_error"),
                "point_in_time_costs": False,
                "note": "Current AoE4 World costs are approximate for historical replays.",
            },
        }
    )
    (output_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    unknown_meta_summary = {k: round(v, 4) for k, v in sorted(unknown_meta_totals.items())}
    report = {
        "selected_matches": len(replays),
        "usable_matches": int(matches_df["usable"].sum()) if not matches_df.empty and "usable" in matches_df else 0,
        "snapshot_rows": snapshot_total,
        "skipped_matches": dict(skip_counts),
        "include_swapped": include_swapped,
        "feature_families": {
            "include_delta": include_delta,
            "include_age_features": include_age_features,
            "include_fractions": include_fractions,
            "include_cancel_eff": include_cancel_eff,
        },
        "row_counts_by_split": dict(row_count_by_split),
        "row_counts_by_minute": dict(row_count_by_minute),
        "row_counts_by_phase": dict(row_count_by_phase),
        "patch_counts": _patch_counts(match_rows),
        "unknown_metadata_totals": unknown_meta_summary,
        "aoe4world_revision": metadata.get("revision"),
        "aoe4world_update_error": metadata.get("update_error"),
        "sample_policy": "deterministic_evenly_spaced_across_replay_id_range_then_split_by_replay_id",
        "end_handling": "snapshots stop after the last observed event; winner labels are resolved from metadata, not inferred from event end",
    }
    (output_dir / "build_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
