#!/bin/bash

# Download TCAD model archive from Zenodo and extract it in the current directory

set -e

ZENODO_RECORD="15868775"
BASE_URL="https://zenodo.org/records/${ZENODO_RECORD}/files"
MODEL_ARCHIVE="models.tar.gz"

# Check dependencies
for cmd in curl tar; do
    if ! command -v $cmd &> /dev/null; then
        echo "Error: '$cmd' is required but not installed"
        exit 1
    fi
done

# Download archive if not already present
if [ -f "$MODEL_ARCHIVE" ]; then
    echo "Skipping download (already exists): $MODEL_ARCHIVE"
else
    echo "Downloading model archive from Zenodo..."
    curl -L --fail --retry 3 --retry-delay 5 \
         --output "$MODEL_ARCHIVE" \
         --progress-bar \
         "${BASE_URL}/${MODEL_ARCHIVE}"

    if [ $? -eq 0 ]; then
        echo "Downloaded: $MODEL_ARCHIVE"
    else
        echo "Download failed"
        exit 1
    fi
fi

# Extract the archive
echo "Extracting archive..."
tar -xzf "$MODEL_ARCHIVE"
echo "Extraction complete. Models are now in: $(pwd)"