import os
import io
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from pathlib import Path
from skimage.color import rgb2lab, lab2rgb

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor



# ------------------------------------------------------------------------------------------------ #
#                                      DATA UTILITY FUNCTIONS                                      #
# ------------------------------------------------------------------------------------------------ #
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
        df_filt = df[df["status"].isin(["2", "3"])]
        imgs_paths = df_filt["image_path"].tolist()

        # Remove duplicates while preserving order
        imgs_paths = list(dict.fromkeys(imgs_paths))

    elif data_type == "png":
        valid_exts = {".png", ".tif", ".tiff"}

        # Sort the images to guarantee deterministic ordering
        imgs_paths = sorted(str(f.resolve()) for f in data_path.iterdir() if f.suffix.lower() in valid_exts)

        # Remove duplicates while preserving order
        imgs_paths = list(dict.fromkeys(imgs_paths))

        # Optional quick image sanity check
        from PIL import Image
        valid_list = []
        for p in imgs_paths:
            try:
                Image.open(p).verify()
                valid_list.append(p)
            except Exception:
                print(f"Warning: Skipping unreadable image: {p}")

        imgs_paths = valid_list

    return imgs_paths, "png"


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


########################################
#   Transform Masks to Polygon String  #
########################################
def masks_to_polygon_string(masks):
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
def find_bigbbox(boxes):
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


# ------------------------------------------------------------------------------------------------ #
#                                    PLOTTING UTILITY FUNCTIONS                                    #
# ------------------------------------------------------------------------------------------------ #
########################################
#           Color Generation           #
########################################
def _generate_colors(n_colors=128, n_samples=5000):
    """
    Generates a perceptually uniform palette of RGB colors.
    Used to color masks and bounding boxes.
    """
    np.random.seed(42)
    rgb = np.random.rand(n_samples, 3)
    lab = rgb2lab(rgb.reshape(1, -1, 3)).reshape(-1, 3)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    kmeans.fit(lab)

    colors_rgb = lab2rgb(kmeans.cluster_centers_.reshape(1, -1, 3)).reshape(-1, 3)
    colors_rgb = np.clip(colors_rgb, 0, 1)
    return colors_rgb

COLORS = _generate_colors()


########################################
def _plot_mask(mask, color="r", ax=None):
    """
    Draws a semi-transparent mask overlay on the given axis
    """
    import numpy as np
    im_h, im_w = mask.shape
    mask_img = np.zeros((im_h, im_w, 4), dtype=np.float32)
    mask_img[..., :3] = color
    mask_img[..., 3] = mask * 0.5

    if ax is None:
        ax = plt.gca()
    ax.imshow(mask_img)


########################################
def _plot_bbox(img_height, img_width, box, box_format="XYXY",
              relative_coords=True, color="r", linestyle="solid",
              text=None, ax=None):
    """
    Draws a bounding box and optional text label
    """
    if box_format == "XYXY":
        x, y, x2, y2 = box
        w = x2 - x
        h = y2 - y
    elif box_format == "XYWH":
        x, y, w, h = box
    elif box_format == "CxCyWH":
        cx, cy, w, h = box
        x = cx - w / 2
        y = cy - h / 2
    else:
        raise RuntimeError(f"Invalid box_format {box_format}")

    if relative_coords:
        x *= img_width
        w *= img_width
        y *= img_height
        h *= img_height

    if ax is None:
        ax = plt.gca()

    rect = patches.Rectangle(
        (x, y), w, h,
        linewidth=1.5, edgecolor=color,
        facecolor="none", linestyle=linestyle
    )
    ax.add_patch(rect)

    if text is not None:
        ax.text(
            x, y - 5, text,
            color=color, weight="bold", fontsize=8,
            bbox={"facecolor": "w", "alpha": 0.75, "pad": 2}
        )

########################################
def plot_results(img, results):
    """
    Creates a matplotlib figure with masks and bounding boxes overlaid.
    Returns the rendered PIL image.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.axis("off")

    nb_objects = len(results["scores"])
    for i in range(nb_objects):
        color = COLORS[i % len(COLORS)]
        mask = results["masks"][i].squeeze(0).cpu().numpy()
        bbox = results["boxes"][i].cpu()

        _plot_mask(mask, color=color, ax=ax)

        w, h = img.size
        prob = results["scores"][i].item()
        _plot_bbox(
            h, w, bbox,
            text=f"(id={i}, prob={prob:.2f})",
            box_format="XYXY",
            color=color,
            relative_coords=False,
            ax=ax,
        )

    # Convert figure to PIL
    fig.canvas.draw()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
    buf.seek(0)

    plt.close(fig)
    return Image.open(buf).convert("RGB")