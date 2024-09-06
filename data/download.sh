#!/bin/bash

# Base URL for the files
BASE_URL="https://huggingface.co/datasets/ankankbhunia/odd-one-out/resolve/main/$1_part_"

# Array of file suffixes
SUFFIXES=("aa" "ab" "ac" "ad" "ae" "af" "ag" "ah" "ai" "aj" "ak" "al" "am" "an" "ao" "ap" "aq" "ar" "as", "at", "au", "av", "aw", "ax", "ay", "az")

# Directory to store downloaded files
DOWNLOAD_DIR="tmp_downloads"
DATASET_PATH="datasets"

mkdir -p $DOWNLOAD_DIR
mkdir -p $DATASET_PATH

# Download each file
for SUFFIX in "${SUFFIXES[@]}"; do
    FILE_URL="${BASE_URL}${SUFFIX}"
    echo  "${DOWNLOAD_DIR}/$1_part_${SUFFIX}" "$FILE_URL"
    wget -O "${DOWNLOAD_DIR}/$1_part_${SUFFIX}" "$FILE_URL"
    if [ $? -ne 0 ]; then
      break
    fi
done

echo "Merging all zip parts..."
# Combine all zip files into one
cat ${DOWNLOAD_DIR}/$1_part_* > ${DOWNLOAD_DIR}/$1.zip

# Extract the combined zip file
unzip ${DOWNLOAD_DIR}/$1.zip -d ${DATASET_PATH}/$1

# Clean up
# rm -rf ${DOWNLOAD_DIR}

echo "$1 - Download and extraction complete."
