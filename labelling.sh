#!/bin/bash

# Load virtualenvwrapper so 'workon' works
source ~/.local/bin/virtualenvwrapper.sh

# --- Require at least 1 argument ---
if [ "$#" -lt 1 ]; then
    echo "Error: Missing required argument."
    echo "Usage: $0 <DATA_PATH> [PROMPT]"
    exit 1
fi

DATAPATH="$1"
PROMPT="${2:-}"   # Empty if not provided

# --- Validate that the data directory exists ---
if [ ! -d "$DATAPATH" ]; then
    echo "Error: DATA_PATH '$DATAPATH' does not exist or is not a directory."
    exit 1
fi

# Stop script on first error
set -e

echo ">>> Running sam3..."
workon sam3-env
cd sam3

# Run sam3 with or without prompt
if [ -n "$PROMPT" ]; then
    python run_sam3.py -d "$DATAPATH" -p "$PROMPT"
else
    python run_sam3.py -d "$DATAPATH"
fi

cd ..

echo ">>> Running sam2..."
workon sam2-env
cd sam2
python run_sam2.py -d "$DATAPATH"
cd ..

echo ">>> Labelling completed successfully!"
