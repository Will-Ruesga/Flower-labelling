import torch

from PIL import Image
from typing import List, Dict, Any

from utils.plot_utils import _plot_results
from utils.parsing_utils import box_xywh_to_cxcywh, normalize_bbox, build_row_dict





# =================================================================================================
#                                         Model Manager Class
# =================================================================================================
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
            cxcywh_tensor = box_xywh_to_cxcywh(xywh_tensor)
            norm_box = normalize_bbox(cxcywh_tensor, width, height).flatten().tolist()

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
            row = build_row_dict(
                image_path=img_abspath,
                num_pages=num_pages,
                page_outputs=page_outputs,
                status_label=status,
                header=header,
            )
            rows.append(row)

        return rows