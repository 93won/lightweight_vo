#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# 1. Check for the destination directory argument.
if [ -z "$1" ]; then
  echo "Error: Please provide a destination directory."
  echo "Usage: $0 <path_to_destination_directory>"
  exit 1
fi

DEST_DIR="$1"

# 2. Create the directory and navigate into it.
mkdir -p "$DEST_DIR"
echo "✅ All datasets will be downloaded and extracted into: ${DEST_DIR}"
cd "$DEST_DIR"


# Base URL for the datasets
BASE_URL="http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset"

# List of datasets to download (by path)
DATASETS=(
    "machine_hall/MH_01_easy/MH_01_easy.zip"
    "machine_hall/MH_02_easy/MH_02_easy.zip"
    "machine_hall/MH_03_medium/MH_03_medium.zip"
    "machine_hall/MH_04_difficult/MH_04_difficult.zip"
    "machine_hall/MH_05_difficult/MH_05_difficult.zip"
    "vicon_room1/V1_01_easy/V1_01_easy.zip"
    "vicon_room1/V1_02_medium/V1_02_medium.zip"
    "vicon_room1/V1_03_difficult/V1_03_difficult.zip"
    "vicon_room2/V2_01_easy/V2_01_easy.zip"
    "vicon_room2/V2_02_medium/V2_02_medium.zip"
    "vicon_room2/V2_03_difficult/V2_03_difficult.zip"
)

# Loop through each dataset in the list
for DATASET_PATH in "${DATASETS[@]}"; do
    
    ZIP_FILE=$(basename "${DATASET_PATH}")
    EXTRACT_DIR="${ZIP_FILE%.zip}"
    
    echo "=================================================="

    # ========================================================== #
    # Added Feature: Skip if Directory Exists (START)
    # ========================================================== #
    # 3. Check if the extraction directory already exists.
    if [ -d "$EXTRACT_DIR" ]; then
        echo "✅ Directory '${EXTRACT_DIR}' already exists. Skipping."
        # 'continue' stops the current iteration and moves to the next item in the loop.
        continue
    fi
    # ========================================================== #
    # Added Feature: Skip if Directory Exists (END)
    # ========================================================== #
    
    echo ">>>>> Processing ${ZIP_FILE}..."
    
    # 4. Download the file using wget.
    echo ">>>>> 1. Downloading..."
    wget -q --show-progress "${BASE_URL}/${DATASET_PATH}"
    
    if [ -f "${ZIP_FILE}" ]; then
        # 5. Create the directory for extraction.
        echo ">>>>> 2. Creating directory '${EXTRACT_DIR}'..."
        mkdir -p "$EXTRACT_DIR"
        
        # 6. Unzip into the new directory (using the -d option).
        echo ">>>>> 3. Extracting into '${EXTRACT_DIR}'..."
        unzip -q "${ZIP_FILE}" -d "$EXTRACT_DIR"
        
        # 7. Delete the original zip file.
        echo ">>>>> 4. Deleting original zip file: ${ZIP_FILE}"
        rm "${ZIP_FILE}"
        
        echo ">>>>> Finished processing ${ZIP_FILE}! 👍"
    else
        echo ">>>>> Error: Failed to download ${ZIP_FILE} ❌"
    fi
    
    echo "" # Add a newline for readability
done

echo "🎉 All tasks have been completed in the '${DEST_DIR}' directory."