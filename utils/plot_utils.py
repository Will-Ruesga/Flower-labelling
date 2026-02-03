import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from pathlib import Path
from skimage.color import rgb2lab, lab2rgb

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor





# =================================================================================================
#                                           PLOTTING UTILS
# =================================================================================================
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
# Plot Image Results
# ---------------------------------------------------------
def render_image_with_mask(image, results):
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
    return _plot_results(image, results)

# ---------------------------------------------------------
def _plot_results(img, results):
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


# ---------------------------------------------------------
# Plot Mask
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
# Plot Bounding Box
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


# =================================================================================================
def load_model(checkpoint_path: Path | str | None, bpe_path: Path | None):
    """
    Load the SAM3 image model and its processor
    
    :param checkpoint_path: Path to model checkpoint (locally not huggingface login)
    :param vocab_path: Path to the BPE vocabulary file. If None, the model is loaded with default settings

    :return Sam3Processor: The processor for the loaded model
    """
    model = build_sam3_image_model(
        bpe_path=str(bpe_path) if bpe_path is not None else None,
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
    )

    return Sam3Processor(model)
