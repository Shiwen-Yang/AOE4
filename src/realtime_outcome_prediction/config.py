from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PARSED_REPLAY_DIR = DATA_DIR / "replays" / "parsed"
RAW_REPLAY_DIR = DATA_DIR / "replays" / "raw"
OUTPUT_DIR = DATA_DIR / "realtime_outcome_prediction"
FEATURE_DIR = OUTPUT_DIR / "features" / "v1"
CACHE_DIR = OUTPUT_DIR / "cache"
DB_PATH = BASE_DIR / "aoe4.duckdb"

SNAPSHOT_MINUTES = 5
DEFAULT_LIMIT = 3_000
DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_VALID_FRAC = 0.15

AOE4_WORLD_BASE_URL = "https://data.aoe4world.com"
AOE4_WORLD_REPO_URL = "https://github.com/aoe4world/data.git"
AOE4_WORLD_REPO_DIR = CACHE_DIR / "aoe4world-data"

PHASES = (
    ("early", 5, 10),
    ("mid", 15, 20),
    ("late", 25, None),
)
