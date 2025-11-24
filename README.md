# Labelling Pipeline

This repository contains a bash script and modified files to run the **SAM2** and **SAM3** pipelines for image segmentation with custom edits.  

> **Note:** This repository does **not include the full SAM2 or SAM3 packages**. You need to download and install them separately.

---

## Repository Structure

```
labelling-repo/
├─ labelling.sh                  # Bash script to run SAM2 and SAM3
├─ sam2_mod/                     # Modified SAM2 files
|  └─ run_sam2.py
|  └─ run_utils.py
├─ sam3_mod/                     # Modified SAM3 files
│  └─ visualization_utils.py     # Edited plot_results function
|  └─ run_sam3.py
|  └─ run_utils.py
├─ README.md
├─ .gitignore
```

- `labelling.sh`: Runs both SAM3 and SAM2 scripts on a dataset folder.  
- `sam3_mod/visualization_utils.py`: Contains modifications to the `plot_results` & `plot_mask` functions.  
- Original `sam2` and `sam3` directories are **not included**. They must be downloaded separately.

---

## Requirements

- **Python 3.12**
- **Virtualenvwrapper** for environment management  
- Installed SAM2 and SAM3 packages in separate directories (each with their own requirements)

---

## Setup

1. Follow the instructions to the of the file `samX-pipsetup-wsl.txt` wehre it explains
how to set up in WSL2 the two environments the virtualwrapper and how to donwload SAM2 and SAM3

2. Copy your modified files from `sam3_mod` into the SAM3 package folder:

```bash
cp sam2_mod/run_utils.py sam2/
cp sam2_mod/run_sam2.py sam2/

cp sam3_mod/visualization_utils.py sam3/sam3/visualization_utils.py
cp sam3_mod/run_utils.py sam3/
cp sam3_mod/run_sam3.py sam3/

```

---

## Usage

Run the pipeline with the bash script:

```bash
bash labelling.sh <DATA_PATH> [PROMPT]
```

- `<DATA_PATH>` → Path to the folder containing images (mandatory)  
- `[PROMPT]` → Optional text prompt for SAM3  

Example:

```bash
bash labelling.sh /mnt/d/Hydrangea2/ "flowers"
```