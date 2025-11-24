import os

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from click_ui import LabellingInteractiveTool

from run_utils import *

########################################
#                 SAM 2                #
########################################
# Set model checkpoint and configuration
checkpoint = "checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

# Init model
device = select_device()
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))

# Load data
args = argument_parse()
data_path = args.dataset_path
img_filenames = load_data(data_path)

# Set mask general path
c_masks_path = os.path.join(data_path, "correct_masks")
i_masks_path = os.path.join(data_path, "incorrect_masks")
os.makedirs(c_masks_path, exist_ok=True)
os.makedirs(i_masks_path, exist_ok=True)

# Initialize dictionary
csv_dict = {
    "image_path": [],
    "mask_path": [],
    "segmentation_status": [],
}

# Iterate over selected rows
for img_fn in img_filenames:

    img_abspath = os.path.join(data_path, img_fn)

    # Launch UI
    ui = LabellingInteractiveTool(predictor=predictor, img_path=img_abspath)

    # Update dict
    csv_dict["image_path"].append(img_abspath)
    csv_dict["mask_path"].append(ui.mask_abspath)
    csv_dict["segmentation_status"].append(ui.decision)

# Update csv with new info
update_csv(csv_dict, data_path, "mask_results.csv")