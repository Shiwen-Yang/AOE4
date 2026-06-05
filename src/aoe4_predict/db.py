import duckdb
import pandas as pd
from pathlib import Path
from .config import DB_PATH

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

PARTICIPANTS_DDL = """
CREATE TABLE IF NOT EXISTS participants (
    game_id                  BIGINT NOT NULL,
    profile_id               BIGINT NOT NULL,
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
    "CREATE INDEX IF NOT EXISTS idx_participants_profile ON participants(profile_id)",
    "CREATE INDEX IF NOT EXISTS idx_participants_started ON participants(game_id)",
    "CREATE INDEX IF NOT EXISTS idx_games_started ON games(started_at)",
    "CREATE INDEX IF NOT EXISTS idx_games_season ON games(season)",
    "CREATE INDEX IF NOT EXISTS idx_games_patch ON games(patch)",
    "CREATE INDEX IF NOT EXISTS idx_games_map ON games(map)",
    "CREATE INDEX IF NOT EXISTS idx_participants_civ ON participants(civilization)",
]


def get_conn(db_path: Path | str | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    path = str(db_path or DB_PATH)
    conn = duckdb.connect(path, read_only=read_only)
    conn.execute("SET threads TO 4")
    conn.execute("SET memory_limit = '8GB'")
    conn.execute("SET temp_directory = '/home/shiwen/tmp/duckdb_spill'")
    return conn


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(GAMES_DDL)
    conn.execute(PARTICIPANTS_DDL)
    for idx_sql in INDEXES:
        conn.execute(idx_sql)


def table_exists(conn: duckdb.DuckDBPyConnection, name: str) -> bool:
    result = conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()
    return result[0] > 0


def row_count(conn: duckdb.DuckDBPyConnection, table: str) -> int:
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def ingest_metadata(conn: duckdb.DuckDBPyConnection, metadata_dir: Path | str) -> dict[str, int]:
    """
    Load map and patch metadata CSVs into DuckDB.

    map_metadata.csv  → table map_metadata
    patch_metadata.csv → table patch_metadata (optional)

    Returns dict of {table_name: rows_loaded}.
    """
    metadata_dir = Path(metadata_dir)
    loaded = {}

    map_csv = metadata_dir / "map_metadata.csv"
    if map_csv.exists():
        df = pd.read_csv(map_csv)
        conn.execute("DROP TABLE IF EXISTS map_metadata")
        conn.register("_map_meta_df", df)
        conn.execute("CREATE TABLE map_metadata AS SELECT * FROM _map_meta_df")
        conn.unregister("_map_meta_df")
        n = row_count(conn, "map_metadata")
        loaded["map_metadata"] = n
        print(f"  map_metadata: {n} rows loaded from {map_csv}")
    else:
        print(f"  Warning: {map_csv} not found — skipping map_metadata")

    patch_csv = metadata_dir / "patch_metadata.csv"
    if patch_csv.exists():
        df = pd.read_csv(patch_csv, parse_dates=["patch_start_at"])
        conn.execute("DROP TABLE IF EXISTS patch_metadata")
        conn.register("_patch_meta_df", df)
        conn.execute("CREATE TABLE patch_metadata AS SELECT * FROM _patch_meta_df")
        conn.unregister("_patch_meta_df")
        n = row_count(conn, "patch_metadata")
        loaded["patch_metadata"] = n
        print(f"  patch_metadata: {n} rows loaded from {patch_csv}")
    else:
        print(f"  Note: {patch_csv} not found — patch features will use DB-derived patch start dates")
        # Derive patch_start_at from game data as fallback
        conn.execute("""
            CREATE OR REPLACE TABLE patch_metadata AS
            SELECT
                patch,
                MIN(started_at) AS patch_start_at,
                NULL::VARCHAR AS patch_name,
                NULL::INTEGER AS season,
                NULL::BOOLEAN AS is_major_patch,
                NULL::BOOLEAN AS is_balance_patch,
                NULL::BOOLEAN AS is_map_pool_patch,
                NULL::BOOLEAN AS is_hotfix,
                'db_derived' AS source_url
            FROM games
            GROUP BY patch
        """)
        n = row_count(conn, "patch_metadata")
        loaded["patch_metadata"] = n
        print(f"  patch_metadata: {n} rows derived from games table (approx start dates)")

    return loaded
