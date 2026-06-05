from __future__ import annotations

from datetime import date, datetime
import hashlib
import random
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import duckdb

from .candidates import game_ids_for_labels
from .config import RAW_REPLAY_DIR, REPLAY_DOWNLOAD_URL


def _today_raw_dir(raw_root: Path = RAW_REPLAY_DIR) -> Path:
    path = raw_root / date.today().isoformat()
    path.mkdir(parents=True, exist_ok=True)
    return path


def choose_profile_id(conn: duckdb.DuckDBPyConnection, game_id: int) -> int | None:
    row = conn.execute(
        """
        SELECT profile_id
        FROM participants
        WHERE game_id = ?
        ORDER BY rating DESC NULLS LAST, profile_id ASC
        LIMIT 1
        """,
        [game_id],
    ).fetchone()
    return int(row[0]) if row else None


def fetch_replay_bytes(game_id: int, profile_id: int, url_template: str = REPLAY_DOWNLOAD_URL) -> bytes:
    url = url_template.format(game_id=game_id, profile_id=profile_id)
    req = Request(url, headers={"User-Agent": "AOE4ReplayHarvest/0.1"})
    with urlopen(req, timeout=120) as resp:
        return resp.read()


def _record_download(
    conn: duckdb.DuckDBPyConnection,
    game_id: int,
    profile_id: int | None,
    raw_path: Path | None,
    status: str,
    sample_group: str,
    size_bytes: int | None = None,
    sha256: str | None = None,
    source: str = "official_replay_endpoint",
    error: str | None = None,
) -> None:
    existing = conn.execute(
        "SELECT attempt_count FROM replay_downloads WHERE game_id = ?",
        [game_id],
    ).fetchone()
    attempt_count = (int(existing[0]) if existing and existing[0] is not None else 0) + 1
    conn.execute("DELETE FROM replay_downloads WHERE game_id = ?", [game_id])
    conn.execute(
        """
        INSERT INTO replay_downloads
            (game_id, profile_id_used, raw_path, download_date, downloaded_at, status,
             size_bytes, sha256, source, sample_group, attempt_count, last_error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            game_id,
            profile_id,
            str(raw_path) if raw_path else None,
            date.today(),
            datetime.utcnow(),
            status,
            size_bytes,
            sha256,
            source,
            sample_group,
            attempt_count,
            error,
        ],
    )


def download_one(
    conn: duckdb.DuckDBPyConnection,
    game_id: int,
    sample_group: str,
    raw_root: Path = RAW_REPLAY_DIR,
    fetcher=fetch_replay_bytes,
) -> str:
    existing = conn.execute(
        "SELECT status FROM replay_downloads WHERE game_id = ?",
        [game_id],
    ).fetchone()
    if existing and existing[0] == "downloaded":
        return "skipped"

    profile_id = choose_profile_id(conn, game_id)
    if profile_id is None:
        _record_download(conn, game_id, None, None, "failed", sample_group, error="no_participant")
        return "failed"

    final_path = _today_raw_dir(raw_root) / f"AgeIV_Replay_{game_id}.gz"
    if final_path.exists() and final_path.stat().st_size > 0:
        data = final_path.read_bytes()
        _record_download(
            conn,
            game_id,
            profile_id,
            final_path,
            "downloaded",
            sample_group,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )
        return "downloaded"

    try:
        data = fetcher(game_id, profile_id)
        if len(data) == 0:
            raise ValueError("empty response")
        if not data.startswith(b"\x1f\x8b"):
            raise ValueError("response is not gzip data")
        tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
        tmp_path.write_bytes(data)
        tmp_path.replace(final_path)
        _record_download(
            conn,
            game_id,
            profile_id,
            final_path,
            "downloaded",
            sample_group,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )
        return "downloaded"
    except HTTPError as exc:
        _record_download(conn, game_id, profile_id, None, "failed", sample_group, error=f"http_{exc.code}")
        if exc.code in {403, 429} or 500 <= exc.code <= 599:
            raise
        return "failed"
    except (URLError, TimeoutError, ValueError, OSError) as exc:
        _record_download(conn, game_id, profile_id, None, "failed", sample_group, error=str(exc))
        return "failed"


def download_group(
    conn: duckdb.DuckDBPyConnection,
    sample_group: str,
    limit: int,
    sleep_min: float,
    sleep_max: float,
    raw_root: Path = RAW_REPLAY_DIR,
    fetcher=fetch_replay_bytes,
) -> dict[str, int]:
    counts = {"downloaded": 0, "failed": 0, "skipped": 0}
    for idx, game_id in enumerate(game_ids_for_labels(conn, sample_group, limit)):
        status = download_one(conn, game_id, sample_group, raw_root=raw_root, fetcher=fetcher)
        counts[status] = counts.get(status, 0) + 1
        if idx < limit - 1:
            time.sleep(random.uniform(sleep_min, sleep_max))
    return counts

