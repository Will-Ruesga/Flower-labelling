from pathlib import Path
from importlib.resources import files, as_file

from img_pred_utils import load_model, data_path_to_img_paths
from applications.task_ui import TaskSelectionUI
from applications.prompt_ui import PromptUI
from applications.automatic_ui import AutomaticUI

# Defines
NONE_B = "-- Select one"
AUTO_B = "Automatic labelling"
PROMPT_B = "Prompt labelling"
BEHAVIORS = (NONE_B, [AUTO_B, PROMPT_B])
CKPT_PATH = "checkpoints/model.safetensors"

# Call Base UI to select the task
taskSelection = TaskSelectionUI(BEHAVIORS)
task = taskSelection.task
data_type = taskSelection.data_type
data_abspath = taskSelection.data_abspath
del taskSelection

# From the data path to list of image paths
data_path = Path(data_abspath)
imgs_paths, mask_format = data_path_to_img_paths(data_path, data_type)

if data_type != ".csv":
    pages_to_label = [0, 1]
else:
    pages_to_label = [0]

# Load model
sam3_root = files("sam3")
bpe_resource = sam3_root / "assets" / "bpe_simple_vocab_16e6.txt.gz"
with as_file(bpe_resource) as bpe_path:
    assert isinstance(bpe_path, Path)
    processor = load_model(CKPT_PATH, bpe_path)

# Set up workspace
# header, corr_path, inc_path = setup_workspace(data_path)
header = ["image_abspath", "csv_abspath", "csv_rel_img_path", "status", "mask_bbox", "mask_rle"]

# Call the appropiate UI depending on the task
if task == BEHAVIORS[1][0]:
    AutomaticUI(processor, imgs_paths, header)
elif task == BEHAVIORS[1][1]:
    PromptUI(processor, imgs_paths, header, pages_to_label)
else:
    exit()