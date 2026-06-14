import duckdb
import pandas as pd
from pathlib import Path

_QUERY = """
SELECT
    g.game_id,
    g.started_at,
    g.season,
    g.map,
    p.profile_id,
    p.civilization,
    p.civilization_randomized,
    p.rating,
    p.mmr
FROM games g
JOIN participants p ON g.game_id = p.game_id
WHERE g.kind = 'rm_1v1'
  AND p.profile_id IS NOT NULL
  AND p.civilization IS NOT NULL
"""


def load_raw(db_path) -> pd.DataFrame:
    conn = duckdb.connect(str(db_path), read_only=True)
    df = conn.execute(_QUERY).df()
    conn.close()
    df["civilization_randomized"] = df["civilization_randomized"].fillna(False)
    return df
