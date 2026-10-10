#!/usr/bin/env bash
#
# author: laquitainesteeve@gmail.com
# execution time: 2 min
# purpose: download and unpack TinyImageNet into data/tiny-imagenet-200/ (~240 MB zip, ~480 MB unpacked).
# usage:
#
#   bash notebooks/download_tiny_imagenet.sh
#
# Output layout:
#
#   data/tiny-imagenet-200/train/<wnid>/images/<image>.JPEG   (100,000 images, stage 3)
#   data/tiny-imagenet-200/val/images/<image>.JPEG            (10,000 images; first 200 sorted = test set)


set -euo pipefail

DATA_DIR="${NEURIPS_DATA_DIR:-data}"
TARGET_DIR="${DATA_DIR}/tiny-imagenet-200"
ZIP_FILE="${DATA_DIR}/tiny-imagenet-200.zip"
# Primary source first, then a mirror serving the identical archive (248100043 bytes).
URLS=(
    "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
    "https://image-net.org/data/tiny-imagenet-200.zip"
)
EXPECTED_BYTES=248100043

echo "=== Downloading & Unpacking TinyImageNet ==="
mkdir -p "$DATA_DIR"

if [ -d "${TARGET_DIR}/train" ] && [ -d "${TARGET_DIR}/val/images" ]; then
    echo "Already present at ${TARGET_DIR}; nothing to do."
else
    echo "[1/3] Downloading (resumes if interrupted, falls back to the mirror)..."
    ok=0
    for URL in "${URLS[@]}"; do
        echo "  trying ${URL}"
        if curl -L --fail --retry 5 --retry-delay 5 --retry-connrefused -C - "$URL" -o "$ZIP_FILE" \
            && [ "$(stat -c %s "$ZIP_FILE")" -eq "$EXPECTED_BYTES" ] \
            && unzip -tq "$ZIP_FILE" > /dev/null 2>&1; then
            ok=1
            break
        fi
        echo "  failed or corrupt; discarding partial file."
        rm -f "$ZIP_FILE"
    done
    if [ "$ok" -ne 1 ]; then
        echo "Error: could not download a valid tiny-imagenet-200.zip from any source."
        exit 1
    fi
    echo "[2/3] Archive verified (size and integrity)."

    echo "[3/3] Unzipping into ${DATA_DIR}..."
    unzip -q -o "$ZIP_FILE" -d "$DATA_DIR"
    rm -f "$ZIP_FILE"
fi

# Sanity check
N_TRAIN=$(find "${TARGET_DIR}/train" -name "*.JPEG" | wc -l)
N_VAL=$(find "${TARGET_DIR}/val/images" -name "*.JPEG" | wc -l)
echo "train images: ${N_TRAIN} (expected 100000)"
echo "val images:   ${N_VAL} (expected 10000)"
if [ "$N_TRAIN" -eq 100000 ] && [ "$N_VAL" -eq 10000 ]; then
    echo "Success! Directory structure verified under ${TARGET_DIR}"
else
    echo "Warning: unexpected image counts."
    exit 1
fi
