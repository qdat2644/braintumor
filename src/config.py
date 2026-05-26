import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = PROJECT_ROOT / "data" / "extracted" / "Training"
TEST_DIR = PROJECT_ROOT / "data" / "extracted" / "Testing"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_MANIFEST = PROCESSED_DIR / "train_manifest.csv"
VAL_MANIFEST = PROCESSED_DIR / "val_manifest.csv"
TEST_MANIFEST = PROCESSED_DIR / "test_manifest.csv"
BINARY_TRAIN_MANIFEST = PROCESSED_DIR / "binary_train_manifest.csv"
BINARY_VAL_MANIFEST = PROCESSED_DIR / "binary_val_manifest.csv"
BINARY_TEST_MANIFEST = PROCESSED_DIR / "binary_test_manifest.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
REPORT_DIR = OUTPUT_DIR / "reports"

MODEL_NAME = os.environ.get("MODEL_NAME", "efficientnet_b0")
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "224"))
USE_PAD_TO_SQUARE = _env_bool("USE_PAD_TO_SQUARE", False)
USE_HORIZONTAL_FLIP = _env_bool("USE_HORIZONTAL_FLIP", True)
USE_CLAHE = _env_bool("USE_CLAHE", False)
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "effb0_manifest_baseline")

BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
RANDOM_SEED = 42
USE_MANIFEST = True
TUMOR_PROB_ALERT_THRESHOLD = 0.20
ENABLE_SAFETY_GATE = True

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
BINARY_CLASS_NAMES = ["tumor", "notumor"]


def as_dict() -> dict[str, object]:
    return {
        "project_root": str(PROJECT_ROOT),
        "train_dir": str(TRAIN_DIR),
        "test_dir": str(TEST_DIR),
        "processed_dir": str(PROCESSED_DIR),
        "train_manifest": str(TRAIN_MANIFEST),
        "val_manifest": str(VAL_MANIFEST),
        "test_manifest": str(TEST_MANIFEST),
        "model_name": MODEL_NAME,
        "use_pad_to_square": USE_PAD_TO_SQUARE,
        "use_horizontal_flip": USE_HORIZONTAL_FLIP,
        "use_clahe": USE_CLAHE,
        "experiment_name": EXPERIMENT_NAME,
        "binary_train_manifest": str(BINARY_TRAIN_MANIFEST),
        "binary_val_manifest": str(BINARY_VAL_MANIFEST),
        "binary_test_manifest": str(BINARY_TEST_MANIFEST),
        "output_dir": str(OUTPUT_DIR),
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "report_dir": str(REPORT_DIR),
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "num_workers": NUM_WORKERS,
        "random_seed": RANDOM_SEED,
        "use_manifest": USE_MANIFEST,
        "tumor_prob_alert_threshold": TUMOR_PROB_ALERT_THRESHOLD,
        "enable_safety_gate": ENABLE_SAFETY_GATE,
        "class_names": CLASS_NAMES,
        "binary_class_names": BINARY_CLASS_NAMES,
    }
