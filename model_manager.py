import torch

from PIL import Image
from pathlib import Path
from typing import List, Dict, Any

from img_pred_utils import masks_to_polygon_string, find_bigbbox, _save_to_csv, plot_results


class ModelManager:
    """
    Manages SAM3 model inference, both single-image and bulk execution
    Keeps UI code clean by encapsulating all model logic here
    """

    def __init__(self, processor):
        self.processor = processor

    ########################################
    #        Single Image Inference        #
    ########################################
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
        
    
    ########################################
    #         Single Page Inference        #
    ########################################
    def _process_page(self, pil_page, prompt, mask_output_type):
        """
        Runs the model on a single page of an image

        :param pil_page: PIL Image object corresponding to one page/frame
        :param prompt: Text prompt provided by the user
        :param mask_output_type: Either "single" or "multiple", controlling mask selection

        :return mask_rle: Polygon-string representation of the mask(s), or "[]" if none
        :return mask_bbox: Bounding box representation as a string, or "[]" if none
        :return has_mask: Boolean indicating whether at least one mask was detected
        """
        output = self.run_model(image=pil_page, prompt=prompt, bbox=None, mask_output_type=mask_output_type)

        # No masks found
        if len(output["scores"]) == 0:
            return "[]", "[]", False

        # Masks present -> convert them
        mask_rle = masks_to_polygon_string(output["masks"])
        mask_bbox = find_bigbbox(output["boxes"])

        return mask_rle, mask_bbox, True
    
    ########################################
    #         Single Page Inference        #
    ########################################
    def _process_image(self, img_path, prompt, mask_output_type):
        """
        Processes an entire image file, which may contain multiple pages (e.g., TIFF)

        :param img_path: Absolute path to the image file
        :param prompt: Text prompt provided by the user
        :param mask_output_type: Either "single" or "multiple", controlling mask selection

        :return mask_rle_pages: List of polygon-string mask representations for each page
        :return mask_bbox_pages: List of bounding-box strings for each page
        :return status: "incorrect" if at least one mask was found across pages, otherwise "discard"
        """
        pages = Image.open(img_path)
        num_pages = getattr(pages, "n_frames", 1)

        mask_rle_pages = []
        mask_bbox_pages = []
        status = "discard"

        # Loops through pages
        for i in range(num_pages):
            pages.seek(i)
            pil_page = pages.copy()

            #  Process page
            mask_rle, mask_bbox, has_mask = self._process_page(pil_page=pil_page, prompt=prompt, mask_output_type=mask_output_type)

            # Append results
            mask_rle_pages.append(mask_rle)
            mask_bbox_pages.append(mask_bbox)

            if has_mask:
                status = "incorrect"

        return mask_rle_pages, mask_bbox_pages, status



    ########################################
    #            Bulk Inference            #
    ########################################
    def run_model_bulk( self, imgs_paths: List[str], prompt: str, header: List[str], 
                       mask_output_type: str = "multiple") -> Dict[str, List[Any]]:
        """
        Runs the model across multiple images and produces a dictionary suitable for CSV output

        :param imgs_paths: List of absolute paths to image files
        :param prompt: Text prompt guiding segmentation
        :param header: List of CSV column names defining output structure
        :param mask_output_type: "single" or "multiple", determining number of masks returned

        :return out_dict: Dictionary containing aggregated segmentation results
                        for all images, ready to be written into a CSV file
        """
        # Prepare output dict (no CSV paths included)
        out_dict = {key: [] for key in header if key not in {"csv_abspath", "csv_rel_img_path"}}

        # Loop thourgh images
        for img_abspath in imgs_paths:

            # Process image fully using helper
            rle_list, bbox_list, status = self._process_image(img_path=img_abspath, prompt=prompt, mask_output_type=mask_output_type)

            # Store results
            out_dict["image_abspath"].append(img_abspath)
            out_dict["mask_rle"].append("[" + ",".join(rle_list) + "]")
            out_dict["mask_bbox"].append("[" + ",".join(bbox_list) + "]")
            out_dict["status"].append(status)

        return out_dict
    

    ########################################
    #             Plot Resutls             #
    ########################################
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
            return image

        # Delegate rendering to plotting helper
        return plot_results(image, results)


    ########################################
    #          Save Results to CSV         #
    ########################################
    def save_results_to_csv(self, out_dict, imgs_paths, header):
        """
        Saves the bulk-inference results into a CSV file

        UI should call this explicitly if needed
        """
        if not imgs_paths:
            return

        first_path = Path(imgs_paths[0])
        _save_to_csv(first_path, out_dict, header)


    def save_single_image(self, img_path, output, correctness_label, header):
        """
        Saves segmentation results for one image into the CSV file.
        This is used by the UI when the user labels image-by-image.

        :param img_path: Path to the image being saved
        :param output: The model output dictionary (scores, boxes, masks, etc.)
        :param correctness_label: "correct", "incorrect", or "discard" depending on UI selection
        :param header: CSV header list defining output order
        """

        # No masks detected
        if len(output["scores"]) == 0:
            mask_rle = "[]"
            mask_bbox = "[]"
            status = "discard"
        else:
            mask_rle = masks_to_polygon_string(output["masks"])
            mask_bbox = find_bigbbox(output["boxes"])
            status = correctness_label   # "1", "2", or "3"

        # Build output dict for ONE row
        out_dict = {key: [] for key in header if key not in {"csv_abspath", "csv_rel_img_path"}}
        out_dict["image_abspath"].append(img_path)
        out_dict["mask_rle"].append("[" + mask_rle + "]")
        out_dict["mask_bbox"].append("[" + mask_bbox + "]")
        out_dict["status"].append(status)

        # Save row
        _save_to_csv(Path(img_path), out_dict, header)



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
