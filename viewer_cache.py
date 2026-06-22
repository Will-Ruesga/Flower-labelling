"""Two-tier in-memory viewer cache (decoded frames + encoded PNGs) with background prefetch.

Workers only prefetch no-overlay PNGs so they never need ``RunContext`` — the
main thread composites any overlay synchronously.
"""

import io
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

import inference


_frames: dict[tuple[int, int], np.ndarray] = {}
_pngs: dict[tuple, bytes] = {}
_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=2)


def get_png(
    image_path: str,
    img_idx: int,
    page_idx: int,
    page_output: dict,
) -> bytes:
    """Return PNG bytes for a page, composing any overlay on cache miss."""
    sig = overlay_sig(page_output)
    key = (img_idx, page_idx, sig)
    with _lock:
        cached = _pngs.get(key)
    if cached is not None:
        return cached

    frame = _get_frame(image_path, img_idx, page_idx)
    rendered = inference.render_image_with_overlay(frame, page_output) if sig is not None else frame

    buf = io.BytesIO()
    rendered.save(buf, format="PNG", compress_level=1)
    data = buf.getvalue()
    with _lock:
        _pngs[key] = data
    return data


def prefetch_image_pages(image_path: str, img_idx: int, num_pages: int) -> None:
    """Warm the no-overlay PNG cache for every page of an image in the background."""
    for page_idx in range(num_pages):
        _executor.submit(_prefetch_worker, image_path, img_idx, page_idx)


def evict_old(keep_img_indices: set[int]) -> None:
    """Drop cache entries whose ``img_idx`` isn't in ``keep_img_indices``."""
    with _lock:
        for key in list(_frames):
            if key[0] not in keep_img_indices:
                del _frames[key]
        for key in list(_pngs):
            if key[0] not in keep_img_indices:
                del _pngs[key]


def clear() -> None:
    """Flush both cache tiers."""
    with _lock:
        _frames.clear()
        _pngs.clear()


def overlay_sig(page_output: dict) -> tuple | None:
    """Hashable key identifying the overlay; ``None`` means no overlay.

    Also used as the ``v=`` token in ``/api/viewer`` URLs so the browser
    cache-busts on overlay change.
    """
    if not page_output or not page_output.get("masks"):
        return None
    scores = page_output.get("scores", [])
    return (len(page_output["masks"]), tuple(round(s, 4) for s in scores))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_frame(image_path: str, img_idx: int, page_idx: int) -> Image.Image:
    """Return the RGB frame, caching in L1. Two threads racing both produce valid output."""
    with _lock:
        arr = _frames.get((img_idx, page_idx))
    if arr is not None:
        return Image.fromarray(arr)

    frame = inference.open_frame(image_path, page_idx)
    with _lock:
        _frames[(img_idx, page_idx)] = np.asarray(frame)
    return frame


def _prefetch_worker(image_path: str, img_idx: int, page_idx: int) -> None:
    """Warm one no-overlay PNG; swallow errors so a crash degrades to a cache miss."""
    try:
        key = (img_idx, page_idx, None)
        with _lock:
            if key in _pngs:
                return
        frame = _get_frame(image_path, img_idx, page_idx)
        buf = io.BytesIO()
        frame.save(buf, format="PNG", compress_level=1)
        with _lock:
            _pngs[key] = buf.getvalue()
    except Exception:
        pass
