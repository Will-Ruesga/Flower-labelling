import os
import pandas as pd

from pathlib import Path





# =================================================================================================
#                                           DATA UTILS
# =================================================================================================
# ---------------------------------------------------------
# Save Rows to CSV
# ---------------------------------------------------------
def save_rows_to_csv(rows: list[dict], header: list[str]):
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
    if first_row.get("fileName"):
        image_path = Path(first_row["fileName"])

    if image_path is None:
        raise ValueError("Rows must include fileName to resolve CSV path.")

    # Output CSV lives next to the images by default
    subfolder = False
    if subfolder:
        csv_path = image_path.parent.parent / "output.csv"
    else:
        csv_path = image_path.parent / "output.csv"

    # Ensure csv path columns are set without mutating input rows
    rows_out = []
    for row in rows:
        row_out = dict(row)
        if "fileName" in header:
            # Store fileName relative to the CSV location
            abs_img_path = row.get("fileName")
            if abs_img_path:
                row_out["fileName"] = os.path.relpath(abs_img_path, csv_path.parent)
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

    if "fileName" in df_old.columns and "fileName" in df_new.columns:
        df_old = df_old[~df_old["fileName"].isin(df_new["fileName"])]

    # Concatenate old + new rows
    df_final = pd.concat([df_old, df_new], ignore_index=True)

    # Save back to disk
    write_csv_with_sep(df_final, csv_path)

    
# ---------------------------------------------------------
# Create/Update & Save the CSV
# ---------------------------------------------------------
def save_to_csv(out_dict: dict, header: list[str]):
    """
    Create or update CSV file and save it

    :param out_dict: Dictionary with the information of all images
    """
    # Convert dict to row dictionaries and delegate to the row-based saver.
    if not out_dict:
        return
    rows = pd.DataFrame(out_dict).to_dict(orient="records")
    save_rows_to_csv(rows, header)


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
        for rel_path in df["fileName"].dropna().tolist():
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
