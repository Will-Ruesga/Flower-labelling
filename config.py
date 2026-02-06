"""
Project-wide configuration constants.
"""

from pathlib import Path

# ---------------------------------------------------------
# CSV schema
# ---------------------------------------------------------
CSV_FILE_COL = "fileName"
CSV_CLASS_COL = "class"
CSV_TARGET_COL = "target"
CSV_SPLIT_COL = "split"
CSV_STATUS_COL = "status"
CSV_COLUMNS = [CSV_FILE_COL, CSV_CLASS_COL, CSV_TARGET_COL, CSV_SPLIT_COL]

# ---------------------------------------------------------
# Metadata filenames
# ---------------------------------------------------------
IMAGE_METADATA_FILENAME = "image_dataset_metadata.csv"
EMBED_METADATA_FILENAME = "embedding_dataset_metadata.csv"

# ---------------------------------------------------------
# Dataset split defaults
# ---------------------------------------------------------
DEFAULT_SPLIT_RATIOS = (0.7, 0.15, 0.15)
DEFAULT_SEED = 42

# ---------------------------------------------------------
# Backbone defaults
# ---------------------------------------------------------
DEFAULT_BACKBONE_REPO = "dinov3/"
DEFAULT_BACKBONE_NAME = "dinov3_vits16"
DEFAULT_BACKBONE_CKPT = "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
DEFAULT_IMAGE_SIZE = 224

# ---------------------------------------------------------
# Training defaults
# ---------------------------------------------------------
DEFAULT_EPOCHS = 30
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 2
DEFAULT_RUNS_DIR = "runs"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_CHECKPOINT_DIR = "checkpoints"

# ---------------------------------------------------------
# UI defaults
# ---------------------------------------------------------
TASK_NONE_LABEL = "-- Select one"
TASK_AUTO_LABEL = "Automatic labelling"
TASK_PROMPT_LABEL = "Prompt labelling"
TASK_BEHAVIORS = (TASK_NONE_LABEL, [TASK_AUTO_LABEL, TASK_PROMPT_LABEL])

PROMPT_UI_BTN_WIDTH = 12
PROMPT_UI_BTN_HEIGHT = 1

# ---------------------------------------------------------
# Model paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SAM3_ROOT = PROJECT_ROOT / "sam3"
BPE_PATH = SAM3_ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz"
CKPT_PATH = PROJECT_ROOT / "checkpoints" / "sam3.pt"

# ---------------------------------------------------------
# Embedding builder defaults
# ---------------------------------------------------------
DEFAULT_EMB_BATCH_SIZE = 32
DEFAULT_EMB_NUM_WORKERS = 4

# ---------------------------------------------------------
# Sweep defaults
# ---------------------------------------------------------
SWEEP_LRS = [1e-4, 3e-4, 1e-3]
SWEEP_BATCHES = [32, 64]
SWEEP_EPOCHS = [30, 50]
SWEEP_SEEDS = [1, 2, 3]
