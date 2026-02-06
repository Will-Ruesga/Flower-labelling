
from config import CSV_FILE_COL, CSV_STATUS_COL
from PIL import Image





# =================================================================================================
#                                           IMAGE UTILS
# =================================================================================================
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
    base_cols = [CSV_FILE_COL, CSV_STATUS_COL]
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
