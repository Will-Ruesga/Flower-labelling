
import sam3

from image_predictor_utils import *
from ui import SAM3UI

# Argparse
args = parse_args()
data_path = args.dataset_path
prompt = args.prompt

# Load model
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
processor = load_model(bpe_path)

# Load data
imgs_path, img_format = load_data(data_path)

# Setup workspace
csv_dict, cor_path, inc_path = setup_workspace(data_path)

# Loop through images in directory
print(f"Images in dataset: {len(imgs_path)}")
for img_fname in imgs_path:

    # Call UI with flags or diff UI
    img_abspath = os.path.join(data_path, img_fname)
    ui = SAM3UI(processor, img_abspath, prompt)

    # Obtain model output
    output, inference_state = ui.process_image()

    # If an object has been located plot and obtain action result from user
    nb_objects = len(output['scores'])
    print(f" - Found {nb_objects} object(s) in image {img_fname}")
    if nb_objects:
        action_result = ui.plot_objects(output)

        # Save masks and store csv data if ordered
        if action_result == "1":
            print("SAVING")
            masks_abspaths = save_masks(output['masks'], img_fname, cor_path)
        elif action_result == "2":
            print("incorrect SAVING")
            masks_abspaths = save_masks(output['masks'], img_fname, inc_path)
        else:
            print("Result not found or discarded")
            masks_abspaths = 0

        # If masks have been saved, update dict
        if len(masks_abspaths):
            print("Inside updating dict")
            for m_abspath in masks_abspaths:
                csv_dict['image_path'].append(img_abspath)
                csv_dict['mask_path'].append(m_abspath)
                csv_dict["action_result"].append(action_result)
    else:
        csv_dict['image_path'].append(img_abspath)
        csv_dict['mask_path'].append(0)
        csv_dict['action_result'].append(3)

# Update csv with masks and info
update_csv(csv_dict, data_path, filename="mask_results.csv")




























































# import os
# import torch

# from PIL import Image
# from sam3.model_builder import build_sam3_image_model
# from sam3.model.sam3_image_processor import Sam3Processor
# from sam3.visualization_utils import plot_results


#     # Image path
#     img_abspath = os.path.join(data_path, img_fn)
#     img = Image.open(img_abspath).convert("RGB")
    
#     # Prompt the model with text
#     inference_state = processor.set_image(img)
#     output = processor.set_text_prompt(state=inference_state, prompt=prompt)

#     # Get the masks, bounding boxes, and scores
#     masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

#     # Only showcase if a flower has been found
#     action = "n"
#     mask_abspath = 0
#     if len(masks):
#         # Get highest score mask
#         best_idx = torch.argmax(scores)
#         best_mask = [masks[best_idx]]
#         best_box = [boxes[best_idx]]
#         best_score = [scores[best_idx]]

#         # Create a small dict to plot only the highest score mask
#         best_output = {"masks": best_mask, "boxes": best_box, "scores": best_score}

#         # Plot resutls
#         action = plot_results(img=img, results=output, skip_box=True)

#         # Save mask if indicated
#         if action == "1":
#             mask_abspath = os.path.join(c_masks_path, img_fn)
#             save_mask(best_mask[0], mask_abspath)
            
#         elif action == "0":
#             mask_abspath = os.path.join(i_masks_path, img_fn)
#             save_mask(best_mask[0], mask_abspath)

#     # Update dictionary regardless of the result
#     csv_dict["image_path"].append(img_abspath)
#     csv_dict["mask_path"].append(mask_abspath)
#     csv_dict["segmentation_status"].append(action)

# # Save values to the csv file
# update_csv(csv_dict, os.path.join(data_path, "mask_results.csv"))

