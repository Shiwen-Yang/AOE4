from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "aoe4.duckdb"
MODEL_DIR = BASE_DIR / "models" / "aoe4_predict"
MODEL_PATH = MODEL_DIR / "lgbm_s10s11s12.txt"
MODEL_META_PATH = MODEL_DIR / "lgbm_s10s11s12_meta.json"
XGB_MODEL_PATH = MODEL_DIR / "xgb_s9s10_test_s11.ubj"
XGB_META_PATH = MODEL_DIR / "xgb_s9s10_test_s11_meta.json"
REPORT_DIR = BASE_DIR / "reports" / "generated"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
QUALITY_REPORT_PATH = REPORT_DIR / "data_quality_report.json"

RM_1V1_KINDS = {"rm_1v1", "rm_solo"}

# Seasons present in data/
ALL_SEASONS = [7, 8, 9, 10, 11, 12]

# Default deployment model: train on recent S10-S12 history.
DEFAULT_TRAIN_SEASONS = [10, 11, 12]

# Temporal split within training seasons
VALID_FRAC = 0.15
TEST_FRAC = 0.15

# Additive smoothing strength (equivalent prior games) for win-rate features
PRIOR_STRENGTH = 10
GLOBAL_WR_PRIOR = 0.5

# Players with fewer than this many prior games are flagged as new/low-history
NEW_PLAYER_THRESHOLD = 10

# Inference-only cold-start priors for players absent from local history.
COLD_START_BASE_SKILL = 1000
COLD_START_OPPONENT_SKILL_GAP = 250
COLD_START_PRIOR_GAMES = 5

# Ingest batch size (games per transaction)
INGEST_BATCH_SIZE = 50_000
