#!/bin/bash

# Green's function and Gradient dataset downloader
# Downloads datasets from Zenodo record 15868775

set -e

ZENODO_RECORD="15868775"
BASE_URL="https://zenodo.org/records/${ZENODO_RECORD}/files"

FILES=("gradZ100k.bin" "greens100k.bin")

# Check if curl is available
if ! command -v curl &> /dev/null; then
    echo "Error: curl is required but not installed"
    exit 1
fi

# Download function
download_file() {
    local filename=$1
    local url="${BASE_URL}/${filename}"
    
    if [ -f "${filename}" ]; then
        echo "Skipping ${filename} (already exists)"
        return 0
    fi
    
    echo "Downloading ${filename}..."
    curl -L --fail --retry 3 --retry-delay 5 \
         --output "${filename}" \
         --progress-bar \
         "${url}"
    
    if [ $? -eq 0 ]; then
        echo "Downloaded: ${filename}"
    else
        echo "Failed to download: ${filename}"
        return 1
    fi
}

# Main execution
echo "Downloading Green's Function & Gradient Dataset"
echo "Files: ${FILES[*]}"
echo "Total size: ~24.8GB"
echo

read -p "Proceed with download? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Download cancelled"
    exit 0
fi

start_time=$(date +%s)

for filename in "${FILES[@]}"; do
    download_file "${filename}"
done

end_time=$(date +%s)
duration=$((end_time - start_time))

echo
echo "Download completed in ${duration} seconds"
echo "Files saved to current directory: $(pwd)"