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
from safetensors.torch import load_file

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ------------------------------------------------------------------------------------------------ #
#                                      DATA UTILITY FUNCTIONS                                      #
# ------------------------------------------------------------------------------------------------ #
# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------
def load_model(checkpoint_path: Path | str, bpe_path: Path | None):
    """
    Load the SAM3 image model and its processor
    
    :param checkpoint_path: Path to model checkpoint (locally not huggingface login)
    :param vocab_path: Path to the BPE vocabulary file. If None, the model is loaded with default settings

    :return Sam3Processor: The processor for the loaded model
    """
    # Check if paths exist
    # if checkpoint_path is None:
    model = build_sam3_image_model(bpe_path=str(bpe_path))
    # checkpoint_path = Path(checkpoint_path)
    # if bpe_path is not None:
    #     if not bpe_path.exists():
    #         raise FileNotFoundError(f"The vocabulary path '{str(bpe_path)}' does not exist.")
    # if not checkpoint_path.exists():
    #     raise FileNotFoundError(f"Checkpoint path '{checkpoint_path}' does not exist.")

    # # Force SAM3 to skip HF download by passing checkpoint_path directly
    # if checkpoint_path.suffix == ".safetensors":
    #     # safetensors requires special loader
    #     model = build_sam3_image_model(bpe_path=str(bpe_path), checkpoint_path=None, load_from_HF=False)
    #     ckpt_data = load_file(checkpoint_path)
    #     sam3_image_ckpt = {k.replace("detector.", ""): v for k, v in ckpt_data.items() if "detector" in k}
    #     if model.inst_interactive_predictor is not None:
    #         sam3_image_ckpt.update(
    #             {
    #                 k.replace("tracker.", "inst_interactive_predictor.model."): v
    #                 for k, v in ckpt_data.items()
    #                 if "tracker" in k
    #             }
    #         )
    #     # Load into model
    #     model.load_state_dict(sam3_image_ckpt, strict=False)
    #     # if len(missing_keys) > 0:
    #     #     print(f"loaded {checkpoint_path} and found missing/unexpected keys:\n{missing_keys=}")
    # else:
    #     raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path.suffix}")

    return Sam3Processor(model)


# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------
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
        df = read_csv_with_sep(data_path)
        # fileName is stored relative to the CSV location
        imgs_paths = []
        for rel_path in df["fileName"].dropna().tolist():
            rel_path = Path(rel_path)
            if rel_path.is_absolute():
                imgs_paths.append(str(rel_path))
            else:
                imgs_paths.append(str((data_path.parent / rel_path).resolve()))

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


# ---------------------------------------------------------
# Filter existing pages
# ---------------------------------------------------------
def filter_existing_pages(pages_to_label: list[int], num_pages: int):
    """
    Filters page indexes, returning only those that exist

    :param pages_to_label: List of integer page indexes to check
    :param num_pages: Number of frames of the image

    :return: List of page indexes that exist in the image
    """
    return [
        page for page in pages_to_label
        if isinstance(page, int) and 0 <= page < num_pages
    ]


# ---------------------------------------------------------
# CSV Helpers
# ---------------------------------------------------------
def ensure_csv_sep_line(csv_path: Path, sep: str = ";") -> None:
    """
    Ensures the first row of the CSV is the Excel separator hint.
    """
    if not csv_path.exists():
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        rest = f.read()

    sep_line = f"sep={sep}"
    if first_line.strip() == sep_line:
        return

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write(sep_line + "\n")
        f.write(first_line)
        f.write(rest)


# ---------------------------------------------------------
def read_csv_with_sep(csv_path: Path, sep: str = ";") -> pd.DataFrame:
    """
    Reads a CSV, adding the sep row if missing.
    """
    ensure_csv_sep_line(csv_path, sep=sep)
    with open(csv_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    if first_line.strip() == f"sep={sep}":
        return pd.read_csv(csv_path, sep=sep, skiprows=1)
    return pd.read_csv(csv_path, sep=sep)


# ---------------------------------------------------------
def write_csv_with_sep(df: pd.DataFrame, csv_path: Path, sep: str = ";") -> None:
    """
    Writes a CSV with the separator hint as the first row.
    """
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write(f"sep={sep}\n")
        df.to_csv(f, index=False, sep=sep)


# ---------------------------------------------------------
# Header + Page Utilities
# ---------------------------------------------------------
def get_num_pages(image_path: str) -> int:
    """
    Gets the number of pages/frames in an image file.
    """
    with Image.open(image_path) as img:
        return getattr(img, "n_frames", 1)


# ---------------------------------------------------------
def build_header_for_pages(num_pages: int) -> list[str]:
    """
    Builds the CSV header with per-page zoom/mask columns.
    """
    base_cols = ["fileName", "status"]
    zoom_cols = []
    mask_cols = []
    for i in range(num_pages):
        zoom_cols.extend([f"ZoomX{i}", f"ZoomY{i}", f"ZoomWidth{i}", f"ZoomHeight{i}"])
        mask_cols.append(f"Mask{i}")
    return base_cols + zoom_cols + mask_cols


# ---------------------------------------------------------
def update_header_for_pages(header: list[str], num_pages: int) -> None:
    """
    Updates the header in-place for a given page count.
    """
    header[:] = build_header_for_pages(num_pages)


# ---------------------------------------------------------
def init_header_from_first_image(paths: list[str], header: list[str]) -> int:
    """
    Initializes the header using the first image in the list.
    """
    num_pages = get_num_pages(paths[0])
    update_header_for_pages(header, num_pages)
    return num_pages


# ---------------------------------------------------------
def filter_paths_by_num_pages(paths: list[str], expected_num_pages: int | None) -> list[str]:
    """
    Keeps only images with the expected number of pages.
    """
    if expected_num_pages is None:
        return paths
    filtered = []
    for p in paths:
        try:
            num_pages = get_num_pages(p)
        except Exception:
            continue

        if num_pages != expected_num_pages:
            print(
                f"[WARNING]: Image {p} has {num_pages} instead of {expected_num_pages} "
                "it might correspond toa different dataset."
            )
            continue

        filtered.append(p)

    return filtered


# ---------------------------------------------------------
# Create/Update & Save the CSV
# ---------------------------------------------------------
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

    # Keep fileName relative to the CSV location
    if "fileName" in df_new.columns:
        df_new["fileName"] = df_new["fileName"].apply(
            lambda p: os.path.relpath(p, csv_path.parent) if p else p
        )

    # Order columns
    df_new = df_new[header]

    if not csv_path.exists():
        write_csv_with_sep(df_new, csv_path)
        return

    # Load old CSV
    df_old = read_csv_with_sep(csv_path)

    if "fileName" in df_old.columns:
        df_old = df_old[~df_old["fileName"].isin(df_new["fileName"])]

    # Concatenate old + new rows
    df_final = pd.concat([df_old, df_new], ignore_index=True)

    # Save back to disk
    write_csv_with_sep(df_final, csv_path)


# ---------------------------------------------------------
# Save Rows to CSV
# ---------------------------------------------------------
def _save_rows_to_csv(rows: list[dict], header: list[str]):
    """
    Create or update CSV file and save it from row dictionaries

    :param rows: List of row dictionaries (one per image)
    :param header: Ordered list of CSV columns
    """
    if not rows:
        return

    # Decide CSV location based on the first row's fileName (expected absolute)
    first_row = rows[0]
    image_path = None
    if first_row.get("fileName"):
        image_path = Path(first_row["fileName"])

    if image_path is None:
        raise ValueError("Rows must include fileName to resolve CSV path.")

    # Output CSV lives next to the images by default
    subfolder = False
    if subfolder:
        csv_path = image_path.parent.parent / "output.csv"
    else:
        csv_path = image_path.parent / "output.csv"

    # Ensure csv path columns are set without mutating input rows
    rows_out = []
    for row in rows:
        row_out = dict(row)
        if "fileName" in header:
            # Store fileName relative to the CSV location
            abs_img_path = row.get("fileName")
            if abs_img_path:
                row_out["fileName"] = os.path.relpath(abs_img_path, csv_path.parent)
        rows_out.append(row_out)

    # Convert rows to DataFrame
    df_new = pd.DataFrame(rows_out)

    # Ensure all header columns exist
    for col in header:
        if col not in df_new.columns:
            df_new[col] = None

    # Order columns
    df_new = df_new[header]

    if not csv_path.exists():
        write_csv_with_sep(df_new, csv_path)
        return

    # Load old CSV
    df_old = read_csv_with_sep(csv_path)

    if "fileName" in df_old.columns and "fileName" in df_new.columns:
        df_old = df_old[~df_old["fileName"].isin(df_new["fileName"])]

    # Concatenate old + new rows
    df_final = pd.concat([df_old, df_new], ignore_index=True)

    # Save back to disk
    write_csv_with_sep(df_final, csv_path)

# ---------------------------------------------------------
# Transform Masks to Polygon String
# ---------------------------------------------------------
def masks_to_polygon_string(masks):
    """
    Convert masks [N, H, W] of ONE PAGE into:
    [
        [ {X:[],Y:[]}, {X:[],Y:[]} ],   # mask of object1
        [ {X:[],Y:[]} ]                 # mask of object2
    ]

    :param masks: Output masks of the model
    """
    # Loop thourhg all masks of 1 page
    mask_strings = []
    for i in range(len(masks)):
        mask_np = torch.squeeze(masks[i]).detach().cpu().numpy()
        mask_np = (mask_np > 0).astype(np.uint8)

        contours, _ = cv2.findContours(mask_np.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # pyright: ignore[reportAttributeAccessIssue]

        contour_strings = []
        for cnt in contours:
            pts = cnt.reshape(-1, 2)
            xs = ",".join(_fmt(x) for x in pts[:, 0])
            ys = ",".join(_fmt(y) for y in pts[:, 1])
            contour_strings.append(f"{{X:[{xs}],Y:[{ys}]}}")

        # One mask = list of polygons
        mask_strings.append(f"[{','.join(contour_strings)}]")

    # The page masks into polygon strings
    return f"[{','.join(mask_strings)}]"

# ---------------------------------------------------------
def _fmt(v: float) -> str:
    """
    Compact float formatting for memory efficiency
    """
    v = float(v)
    # Check if it is 0.0 retrun 0
    if v == 0.0:
        return "0"
    # If integer return only the Real number
    if v.is_integer():
        return str(int(v))
    
    # If float with 0.xxx return .xxx
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    ret = s[1:] if s.startswith("0.") else s
    
    return ret


# ---------------------------------------------------------
# Transform BBoxes to Single Big Box
# ---------------------------------------------------------
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
# ---------------------------------------------------------
# Color Generation
# ---------------------------------------------------------
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


# ---------------------------------------------------------
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


# ---------------------------------------------------------
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

# ---------------------------------------------------------
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
