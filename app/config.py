"""Configuration constants and tunable thresholds."""
from pathlib import Path

# Model
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_VERSION = "distilbert-sst2-v1.0"

# Paths
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REFERENCE_STATS_PATH = MODELS_DIR / "reference_stats.json"
BASELINE_PATH = MODELS_DIR / "baseline.json"

# Event store
DB_PATH = "/tmp/events.db"

# Drift thresholds
DRIFT_LENGTH_KS_P = 0.01
DRIFT_NON_ENGLISH_RATIO = 0.10
DRIFT_JACCARD_MIN = 0.30
DRIFT_LABEL_DELTA = 0.20
DRIFT_WINDOW = 500

# Input validation
MIN_TEXT_LEN = 1
MAX_TEXT_LEN = 2000

# Reference build sample size
REFERENCE_N = 1000
