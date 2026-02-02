from pathlib import Path
from importlib.resources import files, as_file

from img_pred_utils import load_model, data_path_to_img_paths
from applications.task_ui import TaskSelectionUI
from applications.prompt_ui import PromptUI
from applications.automatic_ui import AutomaticUI
from applications.page_selection_ui import PageSelectionUI

# Defines
NONE_B = "-- Select one"
AUTO_B = "Automatic labelling"
PROMPT_B = "Prompt labelling"
BEHAVIORS = (NONE_B, [AUTO_B, PROMPT_B])
PROJECT_ROOT = Path(__file__).resolve().parent
SAM3_ROOT = PROJECT_ROOT / "sam3"
BPE_PATH = SAM3_ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz"
CKPT_PATH = PROJECT_ROOT / "checkpoints" / "model.safetensors"


# ---------------------------------------------------------
# Task Selection
# ---------------------------------------------------------
taskSelection = TaskSelectionUI(BEHAVIORS)
task = taskSelection.task
data_type = taskSelection.data_type
data_abspath = taskSelection.data_abspath
del taskSelection


# ---------------------------------------------------------
# Page Selection
# ---------------------------------------------------------
if data_type != "csv":
    page_selector = PageSelectionUI(max_pages=10)  # open popup
    pages_to_label = page_selector.selected_pages
    del page_selector
else:
    pages_to_label = [0]
print(f"Pages selected: {pages_to_label}")


# ---------------------------------------------------------
# Data Transformation & Model
# ---------------------------------------------------------
# Sanity check
assert BPE_PATH.exists(), f"Missing BPE file: {BPE_PATH}"
assert CKPT_PATH.exists(), f"Missing checkpoint: {CKPT_PATH}"

# Set data to lost of paths
data_path = Path(data_abspath)
imgs_paths, mask_format = data_path_to_img_paths(data_path, data_type)

# Set up workspace
header = ["fileName", "status"]

# Load model
processor = load_model(CKPT_PATH, BPE_PATH)


# ---------------------------------------------------------
# UI State Machine
# ---------------------------------------------------------
if task == BEHAVIORS[1][0]:
    AutomaticUI(processor, imgs_paths, header)  # <- pass pages_to_label
elif task == BEHAVIORS[1][1]:
    PromptUI(processor, imgs_paths, header, pages_to_label)
else:
    exit()
