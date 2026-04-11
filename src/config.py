from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
TABLES_DIR = OUTPUTS_DIR / "tables"
REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR = OUTPUTS_DIR / "models"
HF_CACHE_DIR = MODELS_DIR / "hf_cache"

RAW_REVIEW_FILES = (
    RAW_DIR / "Appliances_5.json.gz",
    RAW_DIR / "Appliances.json.gz",
)
SMALL_RAW_REVIEW_FILE = RAW_DIR / "Appliances_5.json.gz"
LARGE_RAW_REVIEW_FILE = RAW_DIR / "Appliances.json.gz"

DEFAULT_RANDOM_STATE = 42
TEXT_COLUMNS = ("summary", "reviewText")
LABEL_ORDER = ["Negative", "Neutral", "Positive"]
PHASE2_DEV_SAMPLE_SIZE = 60000
PHASE2_LEXICON_COMPARISON_SAMPLE_SIZE = 2000
SECTION16_SUMMARY_COUNT = 10
SECTION16_MIN_WORD_COUNT = 100
SECTION17_QUESTION_PHRASES = (
    "does this",
    "can i",
    "is this",
    "will this",
    "how do i",
    "should i",
    "do i need",
)
RATING_LABEL_MAPPING = {
    "1-2": "Negative",
    "3": "Neutral",
    "4-5": "Positive",
}
