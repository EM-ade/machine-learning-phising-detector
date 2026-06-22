import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATABASE_PATH = os.path.join(BASE_DIR, "phishing_detector.db")

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

COMBINED_DATASET_PATH = os.path.join(DATA_DIR, "combined_dataset.csv")

TFIDF_PARAMS = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "stop_words": "english",
    "strip_accents": "unicode",
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

RF_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 50, 100],
    "min_samples_split": [2, 5, 10],
}

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

INFERENCE_TIMEOUT_MS = 200
MAX_TEXT_LENGTH = 50000

MODEL_VERSION = "1.0.0"
