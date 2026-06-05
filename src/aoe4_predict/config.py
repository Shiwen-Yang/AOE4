from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "aoe4.duckdb"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "lgbm_s9s10_test_s11.txt"
MODEL_META_PATH = MODEL_DIR / "lgbm_s9s10_test_s11_meta.json"
XGB_MODEL_PATH = MODEL_DIR / "xgb_s9s10_test_s11.ubj"
XGB_META_PATH = MODEL_DIR / "xgb_s9s10_test_s11_meta.json"
QUALITY_REPORT_PATH = BASE_DIR / "reports" / "data_quality_report.json"

RM_1V1_KINDS = {"rm_1v1", "rm_solo"}

# Seasons present in data/
ALL_SEASONS = [3, 4, 5, 6, 7, 8, 9, 10, 11]

# Default prototype: train on S10+S11 only (≈22% of data)
DEFAULT_TRAIN_SEASONS = [10, 11]

# Temporal split within training seasons
VALID_FRAC = 0.15
TEST_FRAC = 0.15

# Additive smoothing strength (equivalent prior games) for win-rate features
PRIOR_STRENGTH = 10
GLOBAL_WR_PRIOR = 0.5

# Players with fewer than this many prior games are flagged as new/low-history
NEW_PLAYER_THRESHOLD = 10

# Ingest batch size (games per transaction)
INGEST_BATCH_SIZE = 50_000
