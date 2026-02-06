import os
import re
import pandas as pd

from pathlib import Path

from config import CSV_FILE_COL




# =================================================================================================
#                                           DATA UTILS
# =================================================================================================
# ---------------------------------------------------------
# Save Rows to CSV
# ---------------------------------------------------------
def _slugify_component(value: str, fallback: str) -> str:
    """
    Convert a string into a safe filename component.

    :param value: Raw string to convert
    :param fallback: Fallback string if the result is empty
    :return: Sanitized filename component
    """
    # Normalize and remove path separators
    value = value.strip().lower()
    value = value.replace(os.sep, "-")
    # Replace whitespace with hyphens
    value = re.sub(r"\s+", "-", value)
    # Remove unsafe characters
    value = re.sub(r"[^a-z0-9._-]+", "", value)
    # Trim leftover separators
    value = value.strip("-._")
    return value or fallback


def _format_pages_label(pages_labeled: list[int] | None) -> str:
    """
    Format page indices for filenames using 1-based display.

    :param pages_labeled: List of 0-based page indices
    :return: Page label string for the filename
    """
    if not pages_labeled:
        return "all"
    # De-duplicate and sort, then convert to 1-based for display
    pages_sorted = sorted({int(p) for p in pages_labeled})
    return "-".join(str(p + 1) for p in pages_sorted)


def _build_output_csv_name(
    image_path: Path,
    prompt: str | None,
    mask_output_type: str | None,
    pages_labeled: list[int] | None,
) -> str:
    """
    Build the output CSV filename based on dataset, prompt, pages, and mask type.

    :param image_path: Absolute path of an image in the dataset
    :param prompt: User prompt string
    :param mask_output_type: "single" or "multiple"
    :param pages_labeled: List of 0-based page indices
    :return: Output CSV filename (with .csv extension)
    """
    # Derive dataset name from the parent folder of the image
    dataset_name = _slugify_component(image_path.parent.name, "dataset")
    # Sanitize prompt and mask type for safe filenames
    prompt_name = _slugify_component(prompt or "prompt", "prompt")
    mask_name = _slugify_component(mask_output_type or "mask", "mask")
    # Encode selected pages (1-based for user readability)
    pages_name = _format_pages_label(pages_labeled)
    return f"{dataset_name}_p-{pages_name}_pr-{prompt_name}_tp-{mask_name}.csv"


def save_rows_to_csv(
    rows: list[dict],
    header: list[str],
    prompt: str | None = None,
    mask_output_type: str | None = None,
    pages_labeled: list[int] | None = None,
):
    """
    Create or update CSV file and save it from row dictionaries

    :param rows: List of row dictionaries (one per image)
    :param header: Ordered list of CSV columns
    """
    if not rows:
        return

    # Decide CSV location based on the first row's fileName (expected absolute)
    first_row = rows[0]
    image_path = None
    if first_row.get(CSV_FILE_COL):
        image_path = Path(first_row[CSV_FILE_COL])

    if image_path is None:
        raise ValueError(f"Rows must include {CSV_FILE_COL} to resolve CSV path.")

    # Output CSV lives next to the images by default
    subfolder = False
    csv_name = _build_output_csv_name(
        image_path=image_path,
        prompt=prompt,
        mask_output_type=mask_output_type,
        pages_labeled=pages_labeled,
    )
    if subfolder:
        csv_path = image_path.parent.parent / csv_name
    else:
        csv_path = image_path.parent / csv_name

    # Ensure csv path columns are set without mutating input rows
    rows_out = []
    for row in rows:
        row_out = dict(row)
        if CSV_FILE_COL in header:
            # Store fileName relative to the CSV location
            abs_img_path = row.get(CSV_FILE_COL)
            if abs_img_path:
                row_out[CSV_FILE_COL] = os.path.relpath(abs_img_path, csv_path.parent)
        rows_out.append(row_out)

    # Convert rows to DataFrame
    df_new = pd.DataFrame(rows_out)

    # Ensure all header columns exist
    for col in header:
        if col not in df_new.columns:
            df_new[col] = None

    # Order columns
    df_new = df_new[header]

    if not csv_path.exists():
        write_csv_with_sep(df_new, csv_path)
        return

    # Load old CSV
    df_old = read_csv_with_sep(csv_path)

    if CSV_FILE_COL in df_old.columns and CSV_FILE_COL in df_new.columns:
        df_old = df_old[~df_old[CSV_FILE_COL].isin(df_new[CSV_FILE_COL])]

    # Concatenate old + new rows
    df_final = pd.concat([df_old, df_new], ignore_index=True)

    # Save back to disk
    write_csv_with_sep(df_final, csv_path)

    
# ---------------------------------------------------------
# Create/Update & Save the CSV
# ---------------------------------------------------------
def save_to_csv(
    out_dict: dict,
    header: list[str],
    prompt: str | None = None,
    mask_output_type: str | None = None,
    pages_labeled: list[int] | None = None,
):
    """
    Create or update CSV file and save it

    :param out_dict: Dictionary with the information of all images
    """
    # Convert dict to row dictionaries and delegate to the row-based saver.
    if not out_dict:
        return
    rows = pd.DataFrame(out_dict).to_dict(orient="records")
    save_rows_to_csv(
        rows,
        header,
        prompt=prompt,
        mask_output_type=mask_output_type,
        pages_labeled=pages_labeled,
    )


# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------
def data_path_to_img_paths(data_path: Path, data_type: str | None):
    """
    Given the location of the data (by csv or folder), gets each image absolute path

    :param data_path: Path to the data directory
    :param data_type: String with the type of the data -> ('csv', 'png')

    :return imgs_paths: List of image filenames, or None
    :retrun format: The file extension of the images to save (masks)
    """
    imgs_paths = []
    # If we are working with a csv file
    if data_type == "csv":
        df = read_csv_with_sep(data_path)
        # fileName is stored relative to the CSV location
        imgs_paths = []
        for rel_path in df[CSV_FILE_COL].dropna().tolist():
            rel_path = Path(rel_path)
            if rel_path.is_absolute():
                imgs_paths.append(str(rel_path))
            else:
                imgs_paths.append(str((data_path.parent / rel_path).resolve()))

        # Remove duplicates while preserving order
        imgs_paths = list(dict.fromkeys(imgs_paths))

    elif data_type == "png":
        valid_exts = {".png", ".tif", ".tiff"}

        # Sort the images to guarantee deterministic ordering
        imgs_paths = sorted(str(f.resolve()) for f in data_path.iterdir() if f.suffix.lower() in valid_exts)

        # Remove duplicates while preserving order
        imgs_paths = list(dict.fromkeys(imgs_paths))

        # Optional quick image sanity check
        from PIL import Image
        valid_list = []
        for p in imgs_paths:
            try:
                Image.open(p).verify()
                valid_list.append(p)
            except Exception:
                print(f"Warning: Skipping unreadable image: {p}")

        imgs_paths = valid_list

    return imgs_paths, "png"


# ---------------------------------------------------------
# CSV Helpers
# ---------------------------------------------------------
def ensure_csv_sep_line(csv_path: Path, sep: str = ";") -> None:
    """
    Ensures the first row of the CSV is the Excel separator hint.
    """
    if not csv_path.exists():
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        rest = f.read()

    sep_line = f"sep={sep}"
    if first_line.strip() == sep_line:
        return

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write(sep_line + "\n")
        f.write(first_line)
        f.write(rest)


# ---------------------------------------------------------
def read_csv_with_sep(csv_path: Path, sep: str = ";") -> pd.DataFrame:
    """
    Reads a CSV, adding the sep row if missing.
    """
    ensure_csv_sep_line(csv_path, sep=sep)
    with open(csv_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    if first_line.strip() == f"sep={sep}":
        return pd.read_csv(csv_path, sep=sep, skiprows=1)
    return pd.read_csv(csv_path, sep=sep)


# ---------------------------------------------------------
def write_csv_with_sep(df: pd.DataFrame, csv_path: Path, sep: str = ";") -> None:
    """
    Writes a CSV with the separator hint as the first row.
    """
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write(f"sep={sep}\n")
        df.to_csv(f, index=False, sep=sep)
