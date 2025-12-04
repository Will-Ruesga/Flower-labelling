import os
import cv2
import torch
import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path
from torchvision.transforms.functional import to_pil_image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

########################################
#              Load Model              #
########################################
def load_model(bpe_path: Path | None):
    """
    Load the SAM3 image model and its processor.

    :param vocab_path: Path to the BPE vocabulary file. If None, the model is loaded with default settings.

    :return Sam3Processor: The processor for the loaded model
    """
    # Check if the vocab path exists
    if bpe_path is not None:
        if not bpe_path.exists():
            raise FileNotFoundError(f"The vocabulary path '{str(bpe_path)}' does not exist.")

    # Build the model
    model = build_sam3_image_model(bpe_path=str(bpe_path))
    
    # Return the model processor
    return Sam3Processor(model)


########################################
#               Load Data              #
########################################
def data_path_to_img_paths(data_path: Path, data_type: str | None):
    """
    Given the location of the data (by csv or folder), gets each image absolute path

    :param data_path: Path to the data directory
    :param data_type: String with the type of the data -> ('csv', 'png')

    :return imgs_paths: List of image filenames, or None
    :retrun format: The file extension of the images to save (masks)
    """
    imgs_paths = []
    # If we are working with a csv file
    if data_type == "csv":
        df = pd.read_csv(data_path)
        df_filt = df[df["status"] == "2" and df["status"] == "3"]
        imgs_paths = df_filt["image_path"].tolist()

    elif data_type == "png":
        valid_exts = {".png", ".tif", ".tiff"}
        imgs_paths = [str(f.resolve()) for f in data_path.iterdir() if f.suffix.lower() in valid_exts]
    
    return imgs_paths, "png"


#################################################################################################################
#                                                 SAM3 EXECUTION                                                #
#################################################################################################################
#       Process Images With SAM3       #
########################################
def process_all_images(processor, imgs_paths, prompt, header):
    """
    Processes all the images listes in the imgs_paths variable using the model
    and the inoputs, recieves the output and stores it to the CSV file.

    :param processor: Model processor
    :param imgs_paths: List of absolute paths to the images
    :param prompt: Input prompt for the model
    :param header: First row of the csv (header) which specifies the columns
    
    """
    output_dict = {key: [] for key in header if key not in {"csv_abspath", "csv_rel_img_path"}}
    # Loop throughout images
    for img_abspath in imgs_paths:
        # Initialize
        mask_rle_pages = []
        mask_bbox_pages = []
        status = "3"

        # Open image frames
        pages = Image.open(img_abspath)
        num_pages = getattr(pages, "n_frames", 1)
        
        # Loop throughout the pages
        for i in range(num_pages):
            pages.seek(i)
            image = pages.copy()  # copy the current page

            # Run the model with the current page
            inference_state = processor.set_image(image)
            output = processor.set_text_prompt(state=inference_state, prompt=prompt)

            if len(output["scores"]):
                status = "2"
                mask_rle_pages.append(_masks_to_polygon_string(output["masks"]))
                mask_bbox_pages.append(_find_bigbbox(output["boxes"]))
            else:
                mask_rle_pages.append("[]")
                mask_bbox_pages.append("[]")

        # #### For all pages -> One CSV entry ####
        output_dict["image_abspath"].append(img_abspath)
        output_dict["mask_rle"].append("[" + ",".join(mask_rle_pages) + "]")
        output_dict["mask_bbox"].append("[" + ",".join(mask_bbox_pages) + "]")
        output_dict["status"].append(status)

    # Save/Update CSV
    image_path = Path(imgs_paths[0])
    _save_to_csv(image_path, output_dict, header)


########################################
#   Transform Masks to Polygon String  #
########################################
def _masks_to_polygon_string(masks):
    """
    Convert a list/array/tensor of binary masks [N,H,W]
    into a nested polygon string: [[{X:[..],Y:[..]}],[{..}]]

    :param masks: Ouput masks of the model
    """
    # Loop for each region (mask)
    region = []
    for i in range(len(masks)):
        mask_np = torch.squeeze(masks[0]).detach().cpu().numpy()
        mask_np = (mask_np > 0).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(mask_np.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # pyright: ignore[reportAttributeAccessIssue]

        # Store each contour object as a polygon
        polys = []
        for cnt in contours:
            pts = cnt.reshape(-1, 2)
            xs = (pts[:, 0].astype(float)).tolist()
            ys = (pts[:, 1].astype(float)).tolist()
            polys.append({"X": xs, "Y": ys})

        # Format each polygon as {X:[...], Y:[...]}
        inner = []
        for poly in polys:
            xs = ",".join(f"{float(x):.1f}" for x in poly["X"])
            ys = ",".join(f"{float(y):.1f}" for y in poly["Y"])
            inner.append(f"{{X:[{xs}],Y:[{ys}]}}")
        
        # Append all contours within the region
        region.append(f"[{",".join(inner)}]")

    # Join all different regions (masks)
    return f"[{",".join(region)}]"


########################################
#  Transform BBoxes to Single Big Box  #
########################################
def _find_bigbbox(boxes):
    """
    Given a list of bounding boxes gets a bounding box encompassing all of them

    :param boxes: List of bounding boxes in format XYXY

    :return big_box: Big bounding box that encompasses all the boxes in format XYWH
    """
    # Convert list of boxes to a single tensor of shape (N, 4)
    boxes_tensor = torch.stack([torch.tensor(b) if not isinstance(b, torch.Tensor) else b for b in boxes])

    # Get all x's and y's
    x_mins = boxes_tensor[:, 0]
    y_mins = boxes_tensor[:, 1]
    x_maxs = boxes_tensor[:, 2]
    y_maxs = boxes_tensor[:, 3]

    # Select mins and maxs for big box and transform in XYWH
    big_x = torch.min(x_mins)
    big_y = torch.min(y_mins)
    big_w = torch.max(x_maxs) - big_x
    big_h = torch.max(y_maxs) - big_y

    # Convert to Python floats
    big_box = [big_x.item(), big_y.item(), big_w.item(), big_h.item()]
    return "[" + ",".join(f"{point:.4f}" for point in big_box) + "]"



########################################
#     Create/Update & Save the CSV     #
########################################
def _save_to_csv(image_path: Path, out_dict: dict, header: list[str]):
    """
    Create or update CSV file and save it

    :param image_path: Path to the images
    :param out_dict: Dictionary with the information of all images
    """
    # Decide CSV location
    subfolder = False
    if subfolder:
        csv_path = image_path.parent.parent / "output.csv"
    else:
        csv_path = image_path.parent / "output.csv"

    # Convert dict to DataFrame
    df_new = pd.DataFrame(out_dict)

    # Compute relative path safely
    try:
        csv_rel_img_path = csv_path.relative_to(image_path.parent)
    except ValueError:
        # If not relative, compute a different way
        csv_rel_img_path = os.path.relpath(csv_path, image_path.parent)

    # Add csv_abspath & csv_rel_img_path to the row dict
    df_new["csv_abspath"] = str(csv_path)
    df_new["csv_rel_img_path"] = str(csv_rel_img_path)

    # Order columns
    df_new = df_new[header]

    if not csv_path.exists():
        df_new.to_csv(csv_path, index=False, sep=";")
        return

    # Load old CSV
    df_old = pd.read_csv(csv_path, sep=";")

    if "image_abspath" in df_old.columns:
        df_old = df_old[~df_old["image_abspath"].isin(df_new["image_abspath"])]

    # Concatenate old + new rows
    df_final = pd.concat([df_old, df_new], ignore_index=True)

    # Save back to disk
    df_final.to_csv(csv_path, index=False, sep=";")