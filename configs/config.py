from pathlib import Path
import torch

# =====================================================
# Paths
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"

CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"
OUTPUT_DIR = ROOT / "outputs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Model
# =====================================================

MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"

NUM_CLASSES = 2

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 512

# =====================================================
# Training
# =====================================================

BATCH_SIZE = 8

EPOCHS = 60

LR = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4

PATIENCE = 15

SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_AMP = torch.cuda.is_available()

MAX_GRAD_NORM = 1.0

SAVE_BEST_ONLY = True

T_MAX = EPOCHS