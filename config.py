"""Project-wide constants. No imports from other project modules."""

from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).resolve().parent

SAM3_ROOT: Path = PROJECT_ROOT / "sam3"
BPE_PATH: Path = SAM3_ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz"
CKPT_PATH: Path = PROJECT_ROOT / "checkpoints" / "sam3.pt"

CSV_FILE_COL = "fileName"
CSV_STATUS_COL = "status"
CSV_COLUMNS = [CSV_FILE_COL, CSV_STATUS_COL]
CSV_SEP = ";"

TASK_INCORRECT = "incorrect"
TASK_DISCARD = "discard"
TASK_CORRECT = "correct"
