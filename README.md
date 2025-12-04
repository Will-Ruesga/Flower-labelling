# Labelling Pipeline

This repository contains a bash script and modified files to run the **SAM3** pipeline for image segmentation with custom edits.  

> **Note:** This repository does **not include the full SAM3 package**. You need to download and install it separately.

---

## Repository Structure

```
labelling-repo/
├─ sam3_mod/                    # -- Modified SAM3 files -- #
│  └─ applications
│       └─ task_ui.py           # Task selection UI tool
│       └─ automatic_ui.py      # Automatic labelling UI tool
|  └─ image_predictor.py        # Main script to predict images
|  └─ img_pred_utils.py         # Utility functions
├─ README.md
├─ .gitignore
```
- `sam3_mod`: Directory that contains all the additional files to add to the `sam3` that will be created when downloading SAM3.
- `sam3_mod/image_predictor.py`: Runs the GUIs to execute any type of labelling task
- Original `sam3` directory is **not included**. They must be downloaded separately.

---

## Requirements

- **Python 3.12**
- **Virtualenvwrapper** for environment management  
- Install this package requirements

---

## Setup

1. Follow the instructions to the of the file `samX-pipsetup-wsl.txt` wehre it explains
how to set up in WSL2 the two environments the virtualwrapper and how to donwload SAM3

2. Copy your modified files from `sam3_mod` into the SAM3 package folder. **Not the sam3_mod folder!**

---

## Usage

Run the script image_predictor

```bash
python image_predictor.py
```

A prompt will appear, follow the instructions to label