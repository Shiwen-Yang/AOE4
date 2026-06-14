"""
DuckDB connection + schema for the TEAM match database (aoe4_team.duckdb).

Schema mirrors the 1v1 pipeline but adds a `team_id` column to `participants` so
team membership is preserved (the 1v1 schema assumes exactly 2 players and stores
no team). The 1v1 database is never written here — it is only ATTACHed read-only
for the cross-reference features.
"""
from pathlib import Path

import duckdb

from .config import ONEV1_DB_PATH, TEAM_DB_PATH

DUCKDB_TEMP_DIR = Path("/tmp/duckdb_spill")

GAMES_DDL = """
CREATE TABLE IF NOT EXISTS games (
    game_id     BIGINT PRIMARY KEY,
    started_at  TIMESTAMP,
    finished_at TIMESTAMP,
    duration    INTEGER,
    map_id      BIGINT,
    map         VARCHAR,
    kind        VARCHAR,
    server      VARCHAR,
    patch       VARCHAR,
    season      INTEGER,
    source_file VARCHAR
)
"""

# Same columns as the 1v1 participants table plus team_id (0 / 1 for the two teams).
PARTICIPANTS_DDL = """
CREATE TABLE IF NOT EXISTS participants (
    game_id                  BIGINT NOT NULL,
    profile_id               BIGINT NOT NULL,
    team_id                  INTEGER NOT NULL,
    result                   BOOLEAN,
    civilization             VARCHAR,
    civilization_randomized  BOOLEAN,
    rating                   INTEGER,
    rating_diff              INTEGER,
    mmr                      INTEGER,
    mmr_diff                 INTEGER,
    input_type               VARCHAR,
    PRIMARY KEY (game_id, profile_id)
)
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_tparticipants_profile ON participants(profile_id)",
    "CREATE INDEX IF NOT EXISTS idx_tparticipants_game ON participants(game_id)",
    "CREATE INDEX IF NOT EXISTS idx_tgames_started ON games(started_at)",
    "CREATE INDEX IF NOT EXISTS idx_tgames_season ON games(season)",
    "CREATE INDEX IF NOT EXISTS idx_tgames_kind ON games(kind)",
]


def get_conn(db_path: Path | str | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    path = str(db_path or TEAM_DB_PATH)
    DUCKDB_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(path, read_only=read_only)
    conn.execute("SET threads TO 4")
    conn.execute("SET memory_limit = '8GB'")
    conn.execute(f"SET temp_directory = '{DUCKDB_TEMP_DIR}'")
    return conn


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(GAMES_DDL)
    conn.execute(PARTICIPANTS_DDL)
    for idx_sql in INDEXES:
        conn.execute(idx_sql)


def attach_1v1(conn: duckdb.DuckDBPyConnection, alias: str = "onev1",
               onev1_path: Path | str | None = None) -> bool:
    """
    ATTACH the 1v1 database READ-ONLY under `alias` for cross-reference queries.

    Returns False (and leaves the connection usable) if the 1v1 DB is missing, so
    callers can degrade gracefully to "no 1v1 cross-reference features".
    """
    path = Path(onev1_path or ONEV1_DB_PATH)
    if not path.exists():
        return False
    already = conn.execute(
        "SELECT count(*) FROM duckdb_databases() WHERE database_name = ?", [alias]
    ).fetchone()[0]
    if not already:
        conn.execute(f"ATTACH '{path}' AS {alias} (READ_ONLY)")
    return True


def table_exists(conn: duckdb.DuckDBPyConnection, name: str) -> bool:
    result = conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()
    return result[0] > 0


def row_count(conn: duckdb.DuckDBPyConnection, table: str) -> int:
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
