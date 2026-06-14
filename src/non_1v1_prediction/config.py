from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent

# Separate DB so the 1v1 aoe4.duckdb is never touched by this pipeline.
TEAM_DB_PATH = BASE_DIR / "aoe4_team.duckdb"

# The existing 1v1 database — ATTACHed READ-ONLY for the cross-reference features only.
ONEV1_DB_PATH = BASE_DIR / "aoe4.duckdb"

# Team dumps land here (kept apart from the 1v1 records under data/old_ranked_1v1_records/).
TEAM_DATA_DIR = BASE_DIR / "data" / "team"

MODEL_DIR = BASE_DIR / "models" / "non_1v1_prediction"
REPORT_DIR = BASE_DIR / "reports"

# Mode / season scope. Code is generic over mode so 2v2 can be added later.
DEFAULT_MODE = "rm_4v4"
TEAM_SEASONS = [9, 10, 11]

# players-per-team per mode (used to validate complete teams during ingest)
MODE_TEAM_SIZE = {"rm_4v4": 4, "rm_3v3": 3, "rm_2v2": 2}

# All team modes that feed the teammate co-occurrence network (a premade party plays
# together across modes, so edges are summed over all of these).
TEAM_MODES = ["rm_2v2", "rm_3v3", "rm_4v4"]

# Teammate-network threshold: a pair are "intentional" teammates (an edge) once they have
# played >= TEAMMATE_X games together. Chosen via the opposite-team random baseline; 5 is the
# data-driven default. Weekly snapshots gate premade status on establish_week < match_week.
TEAMMATE_X = 5
WEEK_TRUNC = "week"          # date_trunc granularity for the temporal snapshots

DUMPS_URL = "https://aoe4world.com/dumps"

# Temporal split within the modeling data
VALID_FRAC = 0.15
TEST_FRAC = 0.15

# Additive smoothing strength (equivalent prior games) for win-rate features
PRIOR_STRENGTH = 10
GLOBAL_WR_PRIOR = 0.5

# Players with fewer than this many prior team games are flagged as new/low-history
NEW_PLAYER_THRESHOLD = 10

# Boost/carry exploit thresholds. A "low" player drags down the team mean to game
# matchmaking; a player far below the team mean is a carry-partner candidate.
LOW_MMR_FLOOR = 600          # absolute MMR considered "very low" for a 4v4 stack
CARRY_STD_K = 1.0            # players below (team_mean - k * team_std) count as carried

# "Highly predictable" threshold for the headline (model prob for the favored side)
HIGH_PRED_THRESHOLD = 0.75

# Subgroup analysis: minimum rows for a subgroup to be reported, and skill-band edges
# (on match mean MMR = average of the two teams' mean MMR).
SUBGROUP_MIN_N = 500
SKILL_BANDS = [0, 900, 1100, 1300, 100000]
SKILL_BAND_LABELS = ["<900", "900-1099", "1100-1299", "1300+"]

# Ingest batch size (games per transaction)
INGEST_BATCH_SIZE = 50_000
