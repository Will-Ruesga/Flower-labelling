import torch

from PIL import Image
from typing import List, Dict, Any

from img_pred_utils import masks_to_polygon_string, _save_rows_to_csv, plot_results


class ModelManager:
    """
    Manages SAM3 model inference, both single-image and bulk execution
    Keeps UI code clean by encapsulating all model logic here
    """

    def __init__(self, processor):
        self.processor = processor

    # ---------------------------------------------------------
    # Single Image Inference
    # ---------------------------------------------------------
    def run_model(self, image, prompt, mask_output_type, bbox=None):
        """
        Runs the model on a single image and returns the output dictionary

        :param image: PIL Image (single page)
        :param prompt: Text prompt
        :param mask_output_type: 'single' or 'multiple'
        :param bbox: bounding box in XYWH absolute pixel format, or None

        :return output: Output model dictionary with mask, bbox and score info
        """
        # Set the image in the processor
        inference_state = self.processor.set_image(image)

        # Unpack image size
        width, height = image.size

        # CASE 1: Bounding box + prompt
        if bbox is not None:
            # bbox expected to be [x, y, w, h] in pixel coordinates - convert and normalize
            xywh_tensor = torch.tensor(bbox, dtype=torch.float32).view(-1, 4)
            cxcywh_tensor = _box_xywh_to_cxcywh(xywh_tensor)
            norm_box = _normalize_bbox(cxcywh_tensor, width, height).flatten().tolist()

            # Apply geometric prompt
            self.processor.reset_all_prompts(inference_state)
            inference_state = self.processor.add_geometric_prompt(state=inference_state,box=norm_box,label=True,)

            # Now apply text prompt
            output = self.processor.set_text_prompt(state=inference_state,prompt=prompt)

        # CASE 2: Only text prompt
        else:
            output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)

        # Single mask option
        if mask_output_type == "single" and len(output["scores"]) > 0:
            best_idx = torch.argmax(output["scores"]).item()
            best_output = {}

            indexable_keys = {"scores", "boxes", "masks", "logits"}

            for k, v in output.items():
                if k in indexable_keys:
                    # SAM3 always uses list-like containers for these keys
                    best_output[k] = [v[best_idx]]
                else:
                    # scalar or non-indexed metadata → keep as-is
                    best_output[k] = v

            return best_output
        
        return output
        
    
    # ---------------------------------------------------------
    # Single Page Inference
    # ---------------------------------------------------------
    def _process_page(self, pil_page, prompt, mask_output_type):
        """
        Runs the model on a single page of an image

        :param pil_page: PIL Image object corresponding to one page/frame
        :param prompt: Text prompt provided by the user
        :param mask_output_type: Either "single" or "multiple", controlling mask selection

        :return output: Output model dictionary with mask, bbox and score info
        :return has_mask: Boolean indicating whether at least one mask was detected
        """
        output = self.run_model(image=pil_page, prompt=prompt, bbox=None, mask_output_type=mask_output_type)

        has_mask = len(output["scores"]) > 0
        return output, has_mask
    
    # ---------------------------------------------------------
    # Single Page Inference
    # ---------------------------------------------------------
    def _process_image(self, img_path, prompt, mask_output_type):
        """
        Processes an entire image file, which may contain multiple pages (e.g., TIFF)

        :param img_path: Absolute path to the image file
        :param prompt: Text prompt provided by the user
        :param mask_output_type: Either "single" or "multiple", controlling mask selection

        :return page_outputs: Dict with per-page model outputs
        :return status: "incorrect" if at least one mask was found across pages, otherwise "discard"
        :return num_pages: Number of pages in the image
        """
        status = "discard"
        page_outputs: Dict[int, Any] = {}

        with Image.open(img_path) as pages:
            num_pages = getattr(pages, "n_frames", 1)

            # Loops through pages
            for i in range(num_pages):
                pages.seek(i)
                pil_page = pages.copy()

                # Process page
                output, has_mask = self._process_page(
                    pil_page=pil_page,
                    prompt=prompt,
                    mask_output_type=mask_output_type,
                )

                page_outputs[i] = output

                if has_mask:
                    status = "incorrect"

        return page_outputs, status, num_pages



    # ---------------------------------------------------------
    # Bulk Inference
    # ---------------------------------------------------------
    def run_model_bulk(
        self,
        imgs_paths: List[str],
        prompt: str,
        header: List[str],
        mask_output_type: str = "multiple",
    ) -> List[Dict[str, Any]]:
        """
        Runs the model across multiple images and produces a list of row dictionaries

        :param imgs_paths: List of absolute paths to image files
        :param prompt: Text prompt guiding segmentation
        :param header: List of CSV column names defining output structure
        :param mask_output_type: "single" or "multiple", determining number of masks returned

        :return rows: List of row dictionaries, one per image
        """
        rows: List[Dict[str, Any]] = []

        # Loop thourgh images
        for img_abspath in imgs_paths:

            # Process image fully using helper
            page_outputs, status, num_pages = self._process_image(
                img_path=img_abspath,
                prompt=prompt,
                mask_output_type=mask_output_type,
            )

            # Build single row dict
            row = _build_row_dict(
                image_path=img_abspath,
                num_pages=num_pages,
                page_outputs=page_outputs,
                status_label=status,
                header=header,
            )
            rows.append(row)

        return rows
    

    # ---------------------------------------------------------
    # Plot Resutls
    # ---------------------------------------------------------
    def render_image_with_mask(self, image, results):
        """
        Combines the raw image with mask and bounding box overlays
        Returns the raw image unchanged if no masks were detected

        :param image: PIL Image of the current page/frame
        :param results: Output dictionary from run_model(), containing scores, masks, boxes

        :return overlay_image: PIL Image with overlays applied
        """
        # No masks -> return original image
        if results is None or len(results["scores"]) == 0:
            print(f"Are results none?")
            return image

        # Delegate rendering to plotting helper
        return plot_results(image, results)


    # ---------------------------------------------------------
    # Save Results to CSV
    # ---------------------------------------------------------
    def save_rows_to_csv(self, rows: List[Dict[str, Any]], header: List[str]):
        """
        Saves a list of per-image row dicts into a CSV file
        """
        if not rows:
            return

        _save_rows_to_csv(rows, header)


    def save_results_to_csv(self, rows, header):
        """
        Backwards-compatible wrapper for saving rows to CSV
        """
        self.save_rows_to_csv(rows, header)


    def save_single_image(self, img_path, num_pages, page_outputs, correctness_label, header):
        """
        Saves segmentation results for one image into the CSV file.
        This is used by the UI when the user labels image-by-image.

        :param img_path: Path to the image being saved realative to the csv
        :param num_pages: Number of pages of the current image
        :param page_outputs: The model output dictionary (scores, boxes, masks, etc.) of all pages
        :param correctness_label: "correct", "incorrect", or "discard" depending on UI selection
        :param header: CSV header list defining output order
        """       
        # Build dictionary for 1 row (1 image)
        one_row_dict = _build_row_dict(
            image_path=img_path,
            num_pages=num_pages,
            page_outputs=page_outputs,
            status_label=correctness_label,
            header=header,
        )

        # Save row
        self.save_rows_to_csv([one_row_dict], header)


# ------------------------------------------------------------------------------------------------ #
#                                         HELPER FUNCTIONS                                         #
# ------------------------------------------------------------------------------------------------ #
def _box_xywh_to_cxcywh(box_xywh):
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
def _normalize_bbox(box_cxcywh, width, height):
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
def _get_page_output(page_outputs, page_idx: int):
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
def _big_bbox_xywh(boxes):
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
def _format_bbox_str(bbox_xywh):
    """
    Formats a bbox list into the CSV string representation.
    """
    if not bbox_xywh:
        return "[]"
    return "[" + ",".join(f"{point:.4f}" for point in bbox_xywh) + "]"


# ---------------------------------------------------------
def _build_row_dict(image_path, num_pages: int, page_outputs, status_label: str, header: list[str]) -> dict:
    """
    Given everything known about one image, produce exactly one CSV-ready row dictionary.
    """
    row: Dict[str, Any] = {}
    header_set = set(header)
    img_path_str = str(image_path)

    # Store absolute path first; it gets converted to relative at save-time.
    if "fileName" in header_set:
        row["fileName"] = img_path_str

    for page_idx in range(num_pages):
        output = _get_page_output(page_outputs, page_idx)

        if output is not None and "scores" in output and len(output["scores"]) > 0:
            mask_str_page = masks_to_polygon_string(output["masks"])
            bbox_xywh = _big_bbox_xywh(output["boxes"])
        else:
            mask_str_page = "[[]]"
            bbox_xywh = None
        bbox_str = _format_bbox_str(bbox_xywh)

        mask_key = f"mask{page_idx}"
        bbox_key = f"bbox{page_idx}"
        if mask_key in header_set:
            row[mask_key] = mask_str_page
        if f"Mask{page_idx}" in header_set:
            row[f"Mask{page_idx}"] = mask_str_page
        if bbox_key in header_set:
            row[bbox_key] = bbox_str
        if bbox_xywh is not None:
            zoom_x, zoom_y, zoom_w, zoom_h = (round(v, 4) for v in bbox_xywh)
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
        if f"ZooomHeight{page_idx}" in header_set:
            row[f"ZooomHeight{page_idx}"] = zoom_h

    if "status" in header_set:
        row["status"] = status_label

    return row
