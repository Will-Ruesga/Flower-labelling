import os
import torch
import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from sam3.visualization_utils import plot_results



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



def save_mask(mask, mask_abspath):
    """
    Save a mask tensor or NumPy array as a png.

    Args:
        mask (torch.Tensor or np.ndarray): The mask to save
        mask_abspath (str or Path): Absolute path of where the mask is to be saved

    Returns:
        mask_path (str): Absolute path to the saved mask
    """
    if mask is None:
        return "No mask"
    print(f"mask dtype: {mask.dtype}")
    print(f"mask size: {mask.size}")

    # Ensure NumPy array
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    elif isinstance(mask, np.ndarray):
        mask_np = mask
    else:
        raise TypeError("Mask must be a PyTorch tensor, NumPy array, or None.")

    mask_np = np.squeeze(mask_np)  # collapse extra dims
    mask_img = (mask_np > 0).astype(np.uint8) * 255

    Image.fromarray(mask_img).save(mask_abspath)
    print(f"Mask saved in {mask_abspath}")


def load_data(path):
    """
    Loads image data trying frist with a csv file otherwise with images

    Args:
        path (str): Path to the data
    """
    # Try loading csv file
    img_filenames, _ = load_csv_data(path, "mask_results.csv")

    # If no data available try loading the images in the directory
    if not img_filenames:
        img_filenames = load_image_data(path)
    
    if not img_filenames:
        return None
    else:
        return img_filenames

def load_image_data(path, format=".tiff"):
    """
    Loads the images from a folder

    Args:
        path (str): Path to the data
        format (str): Format of the data
    """
    img_filenames = [f for f in os.listdir(path) if f.lower().endswith(format)]
    if not img_filenames:
        print(f"No {format} files found in directory: {path}")
        return None

    return img_filenames, format


def load_csv_data(path, filename=None):
    """
    Loads CSV data. If `path` is not a CSV file, join it with `filename`
    to construct the full CSV path.

    Args:
        path (str): Either a full CSV file path or a directory.
        filename (str): The CSV filename to use if path is a directory.

    Returns:
        pandas.DataFrame or None
    """
    
    # If path does not end with .csv, join with filename
    if not path.lower().endswith(".csv") and filename is not None:
        path = os.path.join(path, filename)

    # Check if the resulting path exists
    if not os.path.isfile(path):
        return None
    
    # Load CSV
    df = pd.read_csv(path)

    # Filter rows that need labeling
    df_to_label = df[df["segmentation_status"].isin(["s", "n"])]

    # Ensure the column exists
    if "image_path" not in df_to_label.columns:
        print("CSV does not contain required 'image_path' column.")
        return [], ".csv"

    # Extract values
    img_filenames = df_to_label["image_path"].tolist()

    if not img_filenames:
        return None
    else:
        return img_filenames, ""


def update_csv(data_dict, path, filename=None):
    """
    Update the CSV file with new information from data_dict.
    Matches rows by `image_path` and updates `mask_path` and `segmentation_status`.
    
    Args:
        data_dict (dict): Dictionary containing:
            - image_path
            - mask_path
            - segmentation_status
        path (str): Path to CSV file or directory containing it.
        filename (str): CSV filename if path is a directory.
    """

    # Resolve CSV path
    if not path.lower().endswith(".csv") and filename is not None:
        path = os.path.join(path, filename)

    # If sv does not exist create one
    if not os.path.isfile(path):
        df = pd.DataFrame(data_dict)
        df.to_csv(path, index=False)
        return True

    # Load existing CSV
    df = pd.read_csv(path)

    # Create a DataFrame from the small dict
    update_df = pd.DataFrame(data_dict)

    # Ensure required columns exist
    required = {"image_path", "mask_path", "segmentation_status"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {required - set(df.columns)}")

    # Merge using image_path as key
    df = df.merge(update_df, on="image_path", how="left", suffixes=("", "_new"))

    # Update only the rows where new values exist
    for col in ["mask_path", "segmentation_status"]:
        df[col] = df[f"{col}_new"].combine_first(df[col])
        df.drop(columns=[f"{col}_new"], inplace=True)

    # Save back to CSV
    df.to_csv(path, index=False)
    
    return True