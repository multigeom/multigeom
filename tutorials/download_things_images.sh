#!/usr/bin/env bash

# author: laquitainesteeve@gmail.com
# execution time: 2 min

set -e

TARGET_DIR="data/things_images"
OBJECT_IMAGES_DIR="${TARGET_DIR}/object_images"
ZIP_FILE="${TARGET_DIR}/images_THINGS.zip"
PASSWORD_FILE="${TARGET_DIR}/password_images.txt"

echo "=== Downloading & Unpacking THINGS Object Images ==="

mkdir -p "$TARGET_DIR"

# 1. Ensure osfclient is available
python3 -c "import osfclient" 2>/dev/null || pip install osfclient

# 2. Download password file using osfclient
echo "[1/4] Fetching password file..."
osf -p jum2f fetch password_images.txt "$PASSWORD_FILE"
# The file starts with a licence text; the password is the token after "...zip:".
PASSWORD=$(grep -o 'zip: *[^ ]*' "$PASSWORD_FILE" | head -n 1 | sed 's/.*: *//' | tr -d '\r\n')
if [ -z "$PASSWORD" ]; then
    echo "Error: could not parse the password from $PASSWORD_FILE."
    exit 1
fi

# 3. Resolve the direct pre-signed Waterbutler download URL for images_THINGS.zip
echo "[2/4] Resolving direct OSF download link..."
DOWNLOAD_URL=$(python3 -c "
from osfclient import OSF
osf = OSF()
project = osf.project('jum2f')
store = project.storage('osfstorage')
for file_ in store.files:
    if file_.name == 'images_THINGS.zip':
        print(file_._download_url)
        break
")

if [ -z "$DOWNLOAD_URL" ]; then
    echo "Error: Could not resolve download URL for images_THINGS.zip."
    exit 1
fi

# 4. Stream download with automatic resume (-C -) and 10 retries
echo "Downloading images_THINGS.zip (~5.2 GB) with resume support..."
curl -L --fail --retry 10 --retry-delay 5 --retry-connrefused -C - "$DOWNLOAD_URL" -o "$ZIP_FILE"

# 5. Integrity verification
echo "Testing archive integrity..."
if ! unzip -t -P "$PASSWORD" "$ZIP_FILE" > /dev/null 2>&1; then
    echo "Integrity check failed. Resuming download..."
    curl -L --fail --retry 10 --retry-delay 5 -C - "$DOWNLOAD_URL" -o "$ZIP_FILE"
fi

# 6. Extract archive into data/things_images/
echo "[3/4] Unzipping image folders into ${TARGET_DIR}..."
unzip -q -o -P "$PASSWORD" "$ZIP_FILE" -d "$TARGET_DIR"

# 7. Cleanup intermediate zip and password files
echo "[4/4] Cleaning up archives..."
rm -f "$ZIP_FILE" "$PASSWORD_FILE"

# Sanity check
TEST_IMAGE="${OBJECT_IMAGES_DIR}/aardvark/aardvark_01b.jpg"
if [ -f "$TEST_IMAGE" ]; then
    echo "Success! Directory structure verified:"
    echo "  ${TEST_IMAGE}"
else
    echo "Warning: Extraction complete, but $TEST_IMAGE was not found."
fi