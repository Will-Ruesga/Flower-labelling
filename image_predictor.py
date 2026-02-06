from applications.task_ui import TaskSelectionUI
from applications.prompt_ui import PromptUI
from applications.automatic_ui import AutomaticUI
from applications.page_selection_ui import PageSelectionUI

from pathlib import Path

from config import BPE_PATH, CKPT_PATH, CSV_FILE_COL, CSV_STATUS_COL, TASK_BEHAVIORS
from utils.plot_utils import load_model
from utils.data_utils import data_path_to_img_paths





# =================================================================================================
#                                        IMAGE LABELLING TOOL MAIN
# =================================================================================================
# ---------------------------------------------------------
# Task Selection
# ---------------------------------------------------------
taskSelection = TaskSelectionUI(TASK_BEHAVIORS)
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
header = [CSV_FILE_COL, CSV_STATUS_COL]

# Load model
processor = load_model(CKPT_PATH, BPE_PATH)


# ---------------------------------------------------------
# UI State Machine
# ---------------------------------------------------------
if task == TASK_BEHAVIORS[1][0]:
    AutomaticUI(processor, imgs_paths, header)  # <- pass pages_to_label
elif task == TASK_BEHAVIORS[1][1]:
    PromptUI(processor, imgs_paths, header, pages_to_label)
else:
    exit()
