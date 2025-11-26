# import os
# import torch
# import argparse
# import numpy as np
# import pandas as pd

# import matplotlib.pyplot as plt

# from PIL import Image
# from pathlib import Path
# from sam3.visualization_utils import plot_results


########################################
#              Load Model              #
########################################
import os
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def load_model(bpe_path=None):
    """
    Load the SAM3 image model and its processor.

    Args:
        vocab_path (str): Path to the BPE vocabulary file.
                          If None, the model is loaded with default settings.

    Returns:
        Sam3Processor: The processor for the loaded model
    """
    # Check if the vocab path exists
    if bpe_path is not None:
        if not os.path.exists(bpe_path):
            raise FileNotFoundError(f"The vocabulary path '{bpe_path}' does not exist.")

    # Build the model
    model = build_sam3_image_model(bpe_path=bpe_path)
    
    # Return the model processor
    return Sam3Processor(model)


########################################
#               Load Data              #
########################################
import os
import argparse
import pandas as pd
def parse_args():
    """
    Argument parser that get the path to the database and the promt of the model.
    """
    parser = argparse.ArgumentParser(description="Run SAM3 segmentation pipeline.")
    parser.add_argument("-d", "--dataset_path", type=str, required=True,
        help="Path to the folder containing dataset of images.")

    parser.add_argument("-p", "--prompt", type=str, default="flower",
        help="Text prompt used for segmentation (default: 'flower').")
    return parser.parse_args()


def load_data(path, fmt=".tiff"):
    """
    Loads image data, first trying a CSV file and in case of failure tries images

    :param path: Path to the data directory
    :param fmt: String with the image data format

    :return list: List of image filenames, or None
    :retrun fmt: The format of the data
    """
    # First try to load CSV
    csv_file = os.path.join(path, "mask_results.csv")
    img_filenames = []
    if os.path.exists(csv_file):
        try:
            img_filenames, fmt = load_csv_data(path, "mask_results.csv")
        except Exception as e:
            print(f"Warning: Could not load CSV file '{csv_file}': {e}")

    # If CSV failed or no data, load images from directory
    fmt = [".tiff", ".png"] if not fmt else [fmt]

    if not img_filenames:
        try:
            img_filenames, fmt = load_image_data(path, fmt)
        except Exception as e:
            print(f"Warning: Could not load images from '{path}': {e}")

    return img_filenames, fmt


def load_image_data(path, fmts):
    """
    Loads the images from a folder

    :param path: Path to the data
    :param fmts: List of possible formats of the data

    :return img_filenames: List of image filenames, or None
    :return fmt: The format of the data
    """
    # Ensure list
    if isinstance(fmts, str):
        fmts = [fmts]

    # Check directory
    fmt = None
    try:
        all_files = os.listdir(path)
    except Exception as e:
        print(f"Error accessing directory '{path}': {e}")
        return None, fmt

    # Search for files matching each format in order
    img_filenames = []
    for fmt in fmts:
        matching = [f for f in all_files if f.lower().endswith(fmt.lower())]
        if matching:
            img_filenames = matching
            fmt = fmt
            break

    if not img_filenames:
        print(f"No images found in {path} for formats: {fmts}")
        return None, fmt

    return img_filenames, fmt


def load_csv_data(path, filename=None):
    """
    Loads CSV data

    :param path: Either a full CSV file path or a directory
    :param filename: The CSV filename to use if path is a directory

    :return list: List of image filenames, or None
    :return fmt: The format of the data
    """
    fmt = None
    if not path.lower().endswith(".csv") and filename is not None:
        path = os.path.join(path, filename)

    try:
        print(f"cav path: {path}")
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"CSV file not found: {path}")
        return None, fmt
    except Exception as e:
        print(f"Error reading CSV file '{path}': {e}")
        return None, fmt

    # Check required columns
    if not {"action_result", "image_path"}.issubset(df.columns):
        print("CSV missing required columns: 'action_result' and/or 'image_path'.")
        return None, fmt
    df_to_label = df[df["action_result"].isin(["s", "n"])]
    img_filenames = df_to_label["image_path"].tolist()

    if not img_filenames:
        return None, fmt

    # Detect format from first file (keep leading dot)
    _, ext = os.path.splitext(img_filenames[0])
    fmt = ext.lower()

    return img_filenames, fmt


########################################
#            Setup Workspace           #
########################################
def setup_workspace(data_path):
    """
    Setup the workspace by creating directories for the maska and the dictonary to save the information

    :param data_path: Path to the data (csv or images)

    :return csv_dict: Dictionary with the created coluns for storing information
    :return correct_masks_path: Path of the correct masks folder
    :return incorr_masks_path: Path of the incorrect masks folder
    """
    # Create directories
    correct_masks_path = os.path.join(data_path, "correct_masks")
    incorr_masks_path = os.path.join(data_path, "incorrect_masks")
    os.makedirs(correct_masks_path, exist_ok=True)
    os.makedirs(incorr_masks_path, exist_ok=True)

    # Initialize dictionary
    csv_dict = {
        "image_path": [],
        "mask_path": [],
        "action_result": [],
    }

    return csv_dict, correct_masks_path, incorr_masks_path

########################################
#              SAVE MASKS              #
########################################
import torch
from torchvision.transforms.functional import to_pil_image
from PIL import Image
def save_masks(masks, img_fname, mask_path):
    """
    Save all masks for a given image as png as:
        <image_path>/<image_name>_mask_<n>.png

    :param masks: List of masks (tensors)
    :param mask_path: Absolute path of the image where masks should be saved

    :return masks_paths: A list of the absolute masks paths for that image
    """
    nb_masks = len(masks)
    masks_paths = []
    for i in range(nb_masks):
        mask = masks[i]
        # Convert mask to binary uint8 (0 or 255) and save it
        mask_uint8 = (mask > 0).to(torch.uint8) * 255
        mask_img = to_pil_image(mask_uint8)

        img_name, _ = os.path.splitext(img_fname)
        mask_abspath = os.path.join(mask_path, f"{img_name}_mask_{i+1}.png")
        print(mask_abspath)
        mask_img.save(mask_abspath)
        masks_paths.append(mask_abspath)

    return masks_paths


def update_csv(data_dict, path, filename=None):
    """
    Update the CSV file with new information from data_dict.
    Matches rows by `image_path` and updates `mask_path` and `action_result`.

    :param data_dict: Dictionary containing [image_path, mask_path, action_result]
    :param path: Path to CSV file or directory containing it.
    :param filename: CSV filename if path is a directory.
    """

    # Resolve CSV path
    if not path.lower().endswith(".csv") and filename is not None:
        path = os.path.join(path, filename)

    # If csv does not exist create one
    if not os.path.isfile(path):
        df = pd.DataFrame(data_dict)
        df.to_csv(path, index=False)
        return True

    # Load existing CSV
    df = pd.read_csv(path)

    # Create a DataFrame from the small dict
    update_df = pd.DataFrame(data_dict)

    # Ensure required columns exist
    required = {"image_path", "mask_path", "action_result"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {required - set(df.columns)}")

    # Merge using image_path as key
    df = df.merge(update_df, on="image_path", how="left", suffixes=("", "_new"))

    # Update only the rows where new values exist
    for col in ["mask_path", "action_result"]:
        df[col] = df[f"{col}_new"].combine_first(df[col])
        df.drop(columns=[f"{col}_new"], inplace=True)

    # Save back to CSV
    df.to_csv(path, index=False)
    print(f"Results saved in csv: {path}")