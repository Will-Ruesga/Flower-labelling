import os
import torch

from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results

from run_utils import *

# Argparse
args = parse_args()

# Load the model with prompt
model = build_sam3_image_model() #(checkpoint_path="model.safetensors")
processor = Sam3Processor(model)
prompt = args.prompt

# Load images in folder
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

# Loop through images in directory
for img_fn in img_filenames:

    # Image path
    img_abspath = os.path.join(data_path, img_fn)
    img = Image.open(img_abspath).convert("RGB")
    
    # Prompt the model with text
    inference_state = processor.set_image(img)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)

    # Get the masks, bounding boxes, and scores
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

    # Only showcase if a flower has been found
    action = "n"
    mask_abspath = 0
    if len(masks):
        # Get highest score mask
        best_idx = torch.argmax(scores)
        best_mask = [masks[best_idx]]
        best_box = [boxes[best_idx]]
        best_score = [scores[best_idx]]

        # Create a small dict to plot only the highest score mask
        best_output = {"masks": best_mask, "boxes": best_box, "scores": best_score}

        # Plot resutls
        action = plot_results(img=img, results=best_output, skip_box=True)

        # Save mask if indicated
        if action == "1":
            mask_abspath = os.path.join(c_masks_path, img_fn)
            save_mask(best_mask[0], mask_abspath)
            
        elif action == "0":
            mask_abspath = os.path.join(i_masks_path, img_fn)
            save_mask(best_mask[0], mask_abspath)

    # Update dictionary regardless of the result
    csv_dict["image_path"].append(img_abspath)
    csv_dict["mask_path"].append(mask_abspath)
    csv_dict["segmentation_status"].append(action)

# Save values to the csv file
update_csv(csv_dict, os.path.join(data_path, "mask_results.csv"))

