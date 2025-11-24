import os
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def argument_parse():
    parser = argparse.ArgumentParser(description="Run SAM3 segmentation pipeline.")
    parser.add_argument("-d", "--dataset_path", type=str, required=True,
            help="Path to the folder containing dataset of images.")
    return parser.parse_args()


def select_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

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
    df_filt = df[df["segmentation_status"].isin(["s", "n"])]

    # Ensure the column exists
    if "image_path" not in df_filt.columns:
        print("CSV does not contain required 'image_path' column.")
        return [], ".csv"

    # Extract values
    img_filenames = df_filt["image_path"].tolist()

    if not img_filenames:
        return None
    else:
        return img_filenames, df
    

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


# Plot mask
def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)


# Plot image with points defined
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   


# Plot image with bounding box defined
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    


# Plot the multiple masks
def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()