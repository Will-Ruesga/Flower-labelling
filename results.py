"""Row construction and CSV persistence."""

import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from config import CSV_FILE_COL, CSV_SEP, CSV_STATUS_COL
from shared import ModuleResult, Row, SaveReport


def build_row(
    image_path: str,
    num_pages: int,
    page_outputs: list,
    status_label: str,
    header: list[str],
    dataset_root: str,
) -> ModuleResult:
    """Assemble one CSV row from per-page outputs; unlabeled pages become empty cells.

    ``fileName`` is the image path relative to ``dataset_root`` (posix-style),
    so subfolder structure round-trips through the CSV. Only keys in ``header``
    survive so the row matches the canonical schema.
    """
    rel_name = Path(os.path.relpath(image_path, dataset_root)).as_posix()
    row: Row = {
        CSV_FILE_COL: rel_name,
        CSV_STATUS_COL: status_label,
    }
    for i in range(num_pages):
        page = page_outputs[i] if i < len(page_outputs) else None
        if page and page.get("boxes"):
            x, y, w, h = big_bbox_xywh(page["boxes"])
            row[f"ZoomX{i}"] = x
            row[f"ZoomY{i}"] = y
            row[f"ZoomWidth{i}"] = w
            row[f"ZoomHeight{i}"] = h
        else:
            row[f"ZoomX{i}"] = ""
            row[f"ZoomY{i}"] = ""
            row[f"ZoomWidth{i}"] = ""
            row[f"ZoomHeight{i}"] = ""
    for i in range(num_pages):
        page = page_outputs[i] if i < len(page_outputs) else None
        row[f"Mask{i}"] = masks_to_polygon_string(page["masks"]) if page and page.get("masks") else "[[]]"

    row = {k: v for k, v in row.items() if k in header}
    return ModuleResult(ok=True, data=row)


def save_rows(rows: list[Row], save_metadata: dict) -> ModuleResult:
    """Write (or upsert by filename into) the run's CSV file.

    ``save_metadata`` must carry ``{"prompt", "mode", "pages_labeled",
    "dataset_root", "header"}``. Returns a ``SaveReport`` in ``data``.
    """
    header = save_metadata["header"]
    filename = build_output_csv_name(
        save_metadata["prompt"], save_metadata["mode"], save_metadata["pages_labeled"]
    )
    csv_path = Path(save_metadata["dataset_root"]) / filename

    new_df = pd.DataFrame(rows, columns=header)
    if csv_path.exists():
        existing = pd.read_csv(csv_path, sep=CSV_SEP, skiprows=1)
        if CSV_FILE_COL in existing.columns:
            keep = existing[~existing[CSV_FILE_COL].isin(new_df[CSV_FILE_COL])]
            combined = pd.concat([keep, new_df], ignore_index=True)
            combined = combined.reindex(columns=header)
        else:
            combined = new_df  # existing file has a different schema — overwrite
    else:
        combined = new_df

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write(f"sep={CSV_SEP}\n")
        combined.to_csv(f, sep=CSV_SEP, index=False)

    return ModuleResult(
        ok=True,
        data=SaveReport(csv_path=str(csv_path), rows_saved=len(rows)),
    )


def masks_to_polygon_string(masks) -> str:
    """Serialize one page's masks to the ``Mask{i}`` cell string.

    Three nesting levels — page (Mask cell) > object (one detected instance) >
    region (one contour). Shape::

        [ [{X:[…],Y:[…]},{X:[…],Y:[…]}],   ← object 0 — one {X,Y} per region
          [{X:[…],Y:[…]}] ]                ← object 1

    Empty pages serialize as ``"[[]]"``.
    """
    if not masks:
        return "[[]]"
    object_strings: list[str] = []
    for mask in masks:
        mask_arr = (np.asarray(mask) > 0).astype(np.uint8)
        regions, _ = cv2.findContours(mask_arr.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        region_strings: list[str] = []
        for region in regions:
            pts = region.reshape(-1, 2)
            xs = ",".join(_fmt(x) for x in pts[:, 0])
            ys = ",".join(_fmt(y) for y in pts[:, 1])
            region_strings.append(f"{{X:[{xs}],Y:[{ys}]}}")
        object_strings.append(f"[{','.join(region_strings)}]")
    return f"[{','.join(object_strings)}]"


def _fmt(v) -> str:
    """Compact float formatting: ``0.0`` → ``"0"``, ``42.0`` → ``"42"``, ``0.5`` → ``".5"``."""
    v = float(v)
    if v == 0.0:
        return "0"
    if v.is_integer():
        return str(int(v))
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return s[1:] if s.startswith("0.") else s


def big_bbox_xywh(boxes) -> tuple[int, int, int, int]:
    """Return the tight ``(x, y, w, h)`` rectangle around all ``(x, y, w, h)`` boxes."""
    if not boxes:
        return (0, 0, 0, 0)
    xs = [b[0] for b in boxes]
    ys = [b[1] for b in boxes]
    x2s = [b[0] + b[2] for b in boxes]
    y2s = [b[1] + b[3] for b in boxes]
    x, y = min(xs), min(ys)
    return (int(x), int(y), int(max(x2s) - x), int(max(y2s) - y))


def build_output_csv_name(prompt: str, mode: str, pages_labeled: list[int]) -> str:
    """Build ``labels_<slug>_<mode>_p<pages>.csv``; hyphens inside fields, ``_`` between them."""
    prompt_slug = _slugify(prompt, fallback="prompt")
    pages_label = "p" + "-".join(str(p) for p in sorted(pages_labeled)) if pages_labeled else "p"
    return f"labels_{prompt_slug}_{mode}_{pages_label}.csv"


def load_labelled_filenames(dataset_root: str, csv_name: str) -> set[str]:
    """Return the ``fileName`` set already saved in ``dataset_root/csv_name``.

    Empty set if the file is missing or has no ``fileName`` column. Used by
    bulk ops to skip already-labelled images and by the UI progress bar.
    """
    csv_path = Path(dataset_root) / csv_name
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path, sep=CSV_SEP, skiprows=1)
    if CSV_FILE_COL not in df.columns:
        return set()
    return set(df[CSV_FILE_COL].astype(str).tolist())


def _slugify(value: str, fallback: str) -> str:
    """Lowercase + non-alphanumerics→``-``; collapse repeats; empty→fallback."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or fallback
