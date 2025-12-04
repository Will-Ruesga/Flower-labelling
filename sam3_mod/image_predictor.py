import os
import sam3

from pathlib import Path

from img_pred_utils import load_model, data_path_to_img_paths#, setup_workspace
from applications.task_ui import TaskSelectionUI
from applications.automatic_ui import AutomaticUI
from applications.prompt_ui import PromptUI

# Defines
NONE_B = "-- Select one"
AUTO_B = "Automatic labelling"
PROMPT_B = "Prompt only labelling"
BOX_PROMPT_B = "Bounding box and prompt labelling"
BEHAVIORS = (NONE_B, [AUTO_B, PROMPT_B, BOX_PROMPT_B])

# Call Base UI to select the task
taskSelection = TaskSelectionUI(BEHAVIORS)
task = taskSelection.task
data_type = taskSelection.data_type
data_abspath = taskSelection.data_abspath
del taskSelection

# From the data path to list of image paths
data_path = Path(data_abspath)
imgs_paths, mask_format = data_path_to_img_paths(data_path, data_type)

# Load model
sam3_root = Path(sam3.__file__).resolve().parent.parent
bpe_path = sam3_root / "assets" / "bpe_simple_vocab_16e6.txt.gz"
processor = load_model(bpe_path)

# Set up workspace
# header, corr_path, inc_path = setup_workspace(data_path)
header = ["image_abspath", "csv_abspath", "csv_rel_img_path", "status", "mask_bbox", "mask_rle"]

# Call the appropiate UI depending on the task
if task == BEHAVIORS[1][0]:
    print(f"Selected: {task}")
    AutomaticUI(processor, imgs_paths, header)
# elif task == BEHAVIORS[1][1]:
#     print(f"Selected: {task}")
#     PromptUI(processor, imgs_paths, header)

# elif task == BEHAVIORS[1][2]:
#     print(f"Selected: {task}")
# else:
#     exit()