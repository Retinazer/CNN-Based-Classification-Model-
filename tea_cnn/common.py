"""Shared constants and label reconciliation used by load_data.py, train.py, evaluate.py."""
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CACHE_DIR = os.path.join(PROJECT_DIR, "data_cache")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
SAMPLES_DIR = os.path.join(PROJECT_DIR, "samples")
LABELS_PATH = os.path.join(DATA_CACHE_DIR, "labels.json")
BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
HISTORY_CSV_PATH = os.path.join(PROJECT_DIR, "history.csv")

IMG_SIZE = 224  # expected input size for MobileNetV2
SEED = 42

# The two source datasets use different, overlapping naming schemes for the
# same tea-leaf conditions. This mapping normalizes both to one canonical
# label vocabulary so the combined dataset has a single consistent label
# space. Classes with no equivalent in the other dataset are kept as-is.
CANONICAL_LABEL_MAP = {
    # yunusserhat/tea_sickness_dataset raw class names -> canonical name
    "Anthracnose": "anthracnose",
    "algal_leaf": "algal_spot",
    "bird_eye_spot": "bird_eye_spot",
    "brown_blight": "brown_blight",
    "gray_light": "gray_blight",
    "healthy": "healthy",
    "red_leaf_spot": "red_spot",
    "white_spot": "white_spot",
    # Project-AgML/tea_leaf_disease_classification raw class names -> canonical name
    "algal_spot": "algal_spot",
    "gray_blight": "gray_blight",
    "helopeltis": "helopeltis",
    "red_spot": "red_spot",
}


def canonicalize(raw_name: str) -> str:
    """Map a raw label string from either source dataset to its canonical name."""
    if raw_name in CANONICAL_LABEL_MAP:
        return CANONICAL_LABEL_MAP[raw_name]
    # Fall back to a normalized version of whatever we were given so an
    # unexpected label doesn't silently disappear.
    return raw_name.strip().lower().replace(" ", "_")
