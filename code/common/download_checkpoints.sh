#!/bin/bash

# Exit on error
set -e

CHECKPOINT_URL="https://github.com/ai4society/state-centric-gen-planning/releases/download/v1.0.0/checkpoints_v1.zip"
ZIP_NAME="checkpoints_v1.zip"

echo "Downloading pre-trained checkpoints..."

# Check if wget or curl is installed
if command -v wget >/dev/null 2>&1; then
    wget -O $ZIP_NAME $CHECKPOINT_URL
elif command -v curl >/dev/null 2>&1; then
    curl -L -o $ZIP_NAME $CHECKPOINT_URL
else
    echo "Error: Neither wget nor curl found. Please install one to download checkpoints."
    exit 1
fi

echo "Extracting..."
unzip -o $ZIP_NAME
rm $ZIP_NAME

echo "Done! Checkpoints are located in the 'checkpoints/' directory."
