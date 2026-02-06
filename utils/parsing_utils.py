import cv2
import torch
import numpy as np

from typing import Any, Dict

from config import CSV_FILE_COL, CSV_STATUS_COL




# =================================================================================================
#                                           PARSING UTILS
# =================================================================================================
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
# Bounding Box Utils
# ---------------------------------------------------------
def box_xywh_to_cxcywh(box_xywh):
    """
    Converts XYWH format to CxCyWH format (center x, center y, width, height)

    :param box_xywh: bounding box in xywh format

    :return: bounding box in cxcywh format
    """
    x, y, w, h = box_xywh.unbind(-1)
    cx = x + w / 2
    cy = y + h / 2
    return torch.stack([cx, cy, w, h], dim=-1)


# ---------------------------------------------------------
def normalize_bbox(box_cxcywh, width, height):
    """
    Normalizes bbox from absolute pixel coords to [0,1] range

    :param box_cxcywh: bounding box in cxcywh format
    :param width: width of the original image
    :param height: height of the original image

    :return: normalised bounding box
    """
    cx, cy, w, h = box_cxcywh.unbind(-1)

    cx = cx / width
    cy = cy / height
    w = w / width
    h = h / height

    return torch.stack([cx, cy, w, h], dim=-1)


# ---------------------------------------------------------
def get_page_output(page_outputs, page_idx: int):
    """
    Fetches the model output for a given page index from dict or list inputs.
    """
    if isinstance(page_outputs, dict):
        return page_outputs.get(page_idx)
    if isinstance(page_outputs, list):
        if 0 <= page_idx < len(page_outputs):
            return page_outputs[page_idx]
    return None


# ---------------------------------------------------------
def big_bbox_xywh(boxes):
    """
    Computes a single XYWH bbox that encloses all boxes.
    """
    if boxes is None or len(boxes) == 0:
        return None

    boxes_tensor = torch.stack([torch.tensor(b) if not isinstance(b, torch.Tensor) else b for b in boxes])
    x_mins = boxes_tensor[:, 0]
    y_mins = boxes_tensor[:, 1]
    x_maxs = boxes_tensor[:, 2]
    y_maxs = boxes_tensor[:, 3]

    big_x = torch.min(x_mins).item()
    big_y = torch.min(y_mins).item()
    big_w = torch.max(x_maxs).item() - big_x
    big_h = torch.max(y_maxs).item() - big_y

    return [big_x, big_y, big_w, big_h]


# ---------------------------------------------------------
def bbox_to_ints(bbox_xywh):
    """
    Converts bbox values to integer pixel coordinates.
    """
    if not bbox_xywh:
        return None
    return [int(round(float(v))) for v in bbox_xywh]


# ---------------------------------------------------------
def format_bbox_str(bbox_xywh):
    """
    Formats a bbox list into the CSV string representation.
    """
    bbox_ints = bbox_to_ints(bbox_xywh)
    if not bbox_ints:
        return "[]"
    return "[" + ",".join(str(point) for point in bbox_ints) + "]"


# ---------------------------------------------------------
# CSV Row Builder
# ---------------------------------------------------------
def build_row_dict(image_path, num_pages: int, page_outputs, status_label: str, header: list[str]) -> dict:
    """
    Given everything known about one image, produce exactly one CSV-ready row dictionary.
    """
    row: Dict[str, Any] = {}
    header_set = set(header)
    img_path_str = str(image_path)

    # Store absolute path first; it gets converted to relative at save-time.
    if CSV_FILE_COL in header_set:
        row[CSV_FILE_COL] = img_path_str

    for page_idx in range(num_pages):
        output = get_page_output(page_outputs, page_idx)

        if output is not None and "scores" in output and len(output["scores"]) > 0:
            mask_str_page = masks_to_polygon_string(output["masks"])
            bbox_xywh = big_bbox_xywh(output["boxes"])
        else:
            mask_str_page = "[[]]"
            bbox_xywh = None
        bbox_str = format_bbox_str(bbox_xywh)

        mask_key = f"mask{page_idx}"
        bbox_key = f"bbox{page_idx}"
        if mask_key in header_set:
            row[mask_key] = mask_str_page
        if f"Mask{page_idx}" in header_set:
            row[f"Mask{page_idx}"] = mask_str_page
        if bbox_key in header_set:
            row[bbox_key] = bbox_str
        bbox_ints = bbox_to_ints(bbox_xywh)
        if bbox_ints is not None:
            zoom_x, zoom_y, zoom_w, zoom_h = bbox_ints
        else:
            zoom_x = zoom_y = zoom_w = zoom_h = None
        if f"ZoomX{page_idx}" in header_set:
            row[f"ZoomX{page_idx}"] = zoom_x
        if f"ZoomY{page_idx}" in header_set:
            row[f"ZoomY{page_idx}"] = zoom_y
        if f"ZoomWidth{page_idx}" in header_set:
            row[f"ZoomWidth{page_idx}"] = zoom_w
        if f"ZoomHeight{page_idx}" in header_set:
            row[f"ZoomHeight{page_idx}"] = zoom_h

    if CSV_STATUS_COL in header_set:
        row[CSV_STATUS_COL] = status_label

    return row
