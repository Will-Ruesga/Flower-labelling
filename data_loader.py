"""Source validation, dataset/page resolution, and page helpers."""

from pathlib import Path

import pandas as pd
from PIL import Image

from config import CSV_COLUMNS, CSV_FILE_COL, CSV_SEP
from shared import DatasetManifest, ModuleResult


_IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def validate_source(source_type: str, source_path: str) -> ModuleResult:
    """Check that ``source_path`` is a usable source of the declared type."""
    if source_type not in {"csv", "folder"}:
        return ModuleResult(ok=False, message="source_type must be 'csv' or 'folder'")
    if not source_path:
        return ModuleResult(ok=False, message="source path does not exist")
    path = Path(source_path)
    if not path.exists():
        return ModuleResult(ok=False, message="source path does not exist")
    if source_type == "csv":
        if not (path.is_file() and path.suffix.lower() == ".csv"):
            return ModuleResult(ok=False, message="csv source must be a .csv file")
    else:
        if not path.is_dir():
            return ModuleResult(ok=False, message="folder source must be a directory")
    return ModuleResult(ok=True)


def load_dataset(source_type: str, source_path: str) -> ModuleResult:
    """Build a ``DatasetManifest`` from a validated source."""
    path = Path(source_path)
    if source_type == "folder":
        imgs_paths = [
            str(p) for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        ]
        dataset_root = str(path)
    else:
        skiprows = _sep_line_rows(path)
        df = pd.read_csv(path, sep=CSV_SEP, skiprows=skiprows)
        parent = path.parent
        imgs_paths = []
        for value in df[CSV_FILE_COL].tolist():
            value_path = Path(str(value))
            imgs_paths.append(str(value_path if value_path.is_absolute() else parent / value_path))
        dataset_root = str(parent)

    imgs_paths.sort()
    if not imgs_paths:
        return ModuleResult(ok=False, message="no images found in source")
    return ModuleResult(ok=True, data=DatasetManifest(imgs_paths=imgs_paths, dataset_root=dataset_root))


def probe_num_pages(manifest: DatasetManifest) -> ModuleResult:
    """Return the first image's page count, dropping manifest entries that don't match."""
    expected = get_num_pages(manifest.imgs_paths[0])
    manifest.imgs_paths = [p for p in manifest.imgs_paths if get_num_pages(p) == expected]
    return ModuleResult(ok=True, data=expected)


def validate_pages(pages: list[int], expected: int) -> ModuleResult:
    """Ensure every requested page is in ``[0, expected)`` and the list is non-empty."""
    if not pages:
        return ModuleResult(ok=False, message="select at least one page")
    for p in pages:
        if p < 0 or p >= expected:
            return ModuleResult(ok=False, message=f"page {p} out of range (0..{expected - 1})")
    return ModuleResult(ok=True, data=sorted(set(pages)))


def get_num_pages(image_path: str) -> int:
    """Return the frame count (PNG/JPEG always report 1)."""
    with Image.open(image_path) as img:
        return img.n_frames


def build_header_for_pages(num_pages: int) -> list[str]:
    """Return the canonical CSV header: ``fileName, status, Zoom{XYWH}{i}..., Mask{i}...``."""
    header = list(CSV_COLUMNS)
    for i in range(num_pages):
        header.extend([f"ZoomX{i}", f"ZoomY{i}", f"ZoomWidth{i}", f"ZoomHeight{i}"])
    for i in range(num_pages):
        header.append(f"Mask{i}")
    return header


def _sep_line_rows(csv_path: Path) -> int:
    """Return 1 if the file starts with the ``sep=;`` Excel hint, else 0."""
    with csv_path.open("r", encoding="utf-8") as f:
        first = f.readline()
    return 1 if first.startswith("sep=") else 0
