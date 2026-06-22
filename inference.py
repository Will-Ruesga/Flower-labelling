"""SAM3 lifecycle, per-page inference, and mask overlay rendering."""

import threading

import numpy as np
import torch
from PIL import Image, ImageDraw

from config import BPE_PATH, CKPT_PATH
from shared import ModuleResult


_processor = None
# Flask dev server is multi-threaded; serialise SAM3 so two clicks can't collide.
_inference_lock = threading.Lock()


_COLORS = [
    (255, 64, 64),
    (64, 200, 64),
    (64, 128, 255),
    (240, 200, 0),
    (200, 80, 220),
    (0, 200, 200),
    (255, 140, 0),
    (130, 90, 240),
]
_ALPHA = 110


def ensure_model_ready() -> ModuleResult:
    """Build the SAM3 model from local weights on first call; no-op afterwards."""
    global _processor
    if _processor is not None:
        return ModuleResult(ok=True)
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(
        bpe_path=str(BPE_PATH),
        checkpoint_path=str(CKPT_PATH),
        load_from_HF=False,
        device=device,
    )
    _processor = Sam3Processor(model, device=device)
    return ModuleResult(ok=True)


def run_page(
    image_path: str,
    page_index: int,
    prompt: str,
    mode: str,
    threshold: float,
    *,
    cached_state: dict | None = None,
) -> ModuleResult:
    """Run SAM3 on one page. With ``cached_state``, skips the backbone.

    Returns ``data=(output_dict, state_cpu)`` where ``output_dict`` has
    ``{"masks", "boxes", "scores"}`` and ``state_cpu`` is the full state with
    tensors on CPU for later reuse.
    """
    with _inference_lock:
        if cached_state is not None:
            state = _state_to_device(cached_state)
        else:
            frame = open_frame(image_path, page_index)
            state = _processor.set_image(frame)
        _processor.confidence_threshold = threshold
        state = _processor.set_text_prompt(prompt, state)
        output = _extract_output(state, mode)
        state_cpu = _state_to_cpu(state)
    return ModuleResult(ok=True, data=(output, state_cpu))


def rerun_with_threshold(state_cpu: dict, threshold: float, mode: str) -> ModuleResult:
    """Reapply the prediction heads to a cached state with a new threshold.

    Skips backbone and text encoder. Returns the same
    ``(output_dict, state_cpu)`` shape as ``run_page``.
    """
    with _inference_lock:
        state = _state_to_device(state_cpu)
        state = _processor.set_confidence_threshold(threshold, state)
        output = _extract_output(state, mode)
        state_cpu_new = _state_to_cpu(state)
    return ModuleResult(ok=True, data=(output, state_cpu_new))


def render_image_with_overlay(image: Image.Image, page_output: dict) -> Image.Image:
    """Return ``image`` with masks, bboxes, and scores painted on."""
    if not page_output or not page_output.get("masks"):
        return image
    base = image.convert("RGBA")

    masks = page_output["masks"]
    boxes = page_output.get("boxes", [])
    scores = page_output.get("scores", [])

    H, W = np.asarray(masks[0]).shape
    overlay = np.zeros((H, W, 4), dtype=np.uint8)
    for i, mask in enumerate(masks):
        overlay[np.asarray(mask, dtype=bool)] = (*_COLORS[i % len(_COLORS)], _ALPHA)
    base = Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA"))

    draw = ImageDraw.Draw(base)
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x, y, w, h = box
        color = _COLORS[i % len(_COLORS)] + (255,)
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        draw.text((x + 3, max(0, y - 14)), f"{score:.2f}", fill=color)

    return base.convert("RGB")


def open_frame(image_path: str, page_index: int) -> Image.Image:
    """Open an image and return one page as an independent RGB ``Image``."""
    img = Image.open(image_path)
    if page_index > 0:
        img.seek(page_index)
    return img.copy().convert("RGB")


def _extract_output(state: dict, mode: str) -> dict:
    """Pull masks/boxes/scores off a SAM3 state dict into CPU numpy/lists."""
    masks = [m.squeeze(0).cpu().numpy() for m in state["masks"]]
    boxes = [
        (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        for x1, y1, x2, y2 in state["boxes"].cpu().numpy()
    ]
    scores = state["scores"].cpu().tolist()

    if mode == "single" and scores:
        best = int(np.argmax(scores))
        return {"masks": [masks[best]], "boxes": [boxes[best]], "scores": [scores[best]]}
    return {"masks": masks, "boxes": boxes, "scores": scores}


def _state_to_cpu(state: dict) -> dict:
    """Move every tensor in ``state`` to CPU."""
    return _move_tensors(state, lambda t: t.cpu())


def _state_to_device(state: dict) -> dict:
    """Move every tensor in ``state`` onto the processor's device."""
    return _move_tensors(state, lambda t: t.to(_processor.device))


def _move_tensors(obj, fn):
    """Walk a nested dict/list/tuple and apply ``fn`` to every tensor."""
    if isinstance(obj, dict):
        return {k: _move_tensors(v, fn) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_tensors(v, fn) for v in obj)
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    return obj
