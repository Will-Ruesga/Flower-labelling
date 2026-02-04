# Labelling Pipeline

This repository contains a bash script and modified files to run the **SAM3** pipeline for image segmentation with custom edits.  

> **Note:** This repository does **not include the full SAM3 package**. You need to download and install it separately.

---

## Repository Structure

```
labelling-repo/
├─ sam3/                    # -- SAM3 repository [Not included] -- #
├─ applications
│   └─ task_ui.py           # Task selection UI tool
│   └─ automatic_ui.py      # Automatic labelling UI tool
│   └─ prompt_ui.py         # Prompt labelling UI tool
├─ image_predictor.py       # Main script to predict images
├─ utils                    # Utility functions
│   └─ data_utils.py
│   └─ image_utils.py
│   └─ parsing_utils.py
│   └─ plot_utils.py
├─ README.md
├─ requiremtns.txt          # Requirements
├─ wsl_sam3_setup.txt       # Set up instructions for WSL and SAM3
```

---

## Setup

1. Follow the instructions to the of the file `wsl_sam3_setup.txt` wehre it explains
how to set up WSL and SAM3.

2. Install the requirements in `requiremtns.txt`.

---

## Usage

Run the script image_predictor

```bash
python image_predictor.py
```

A prompt will appear, follow the instructions to label the images