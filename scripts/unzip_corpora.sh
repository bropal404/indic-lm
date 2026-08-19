#!/usr/bin/env bash
set -euo pipefail

# Script to unpack heavy dataset archive to the exact designated directories
# Usage: ./scripts/unzip_corpora.sh [archive_path]

ARCHIVE_PATH="${1:-corpora_dataset_bundle.tar.gz}"

if [ ! -f "$ARCHIVE_PATH" ]; then
    echo "Error: Archive file '$ARCHIVE_PATH' not found."
    echo "Usage: $0 [path_to_corpora_dataset_bundle.tar.gz]"
    exit 1
fi

echo "============================================================"
echo "Unpacking Heavy Corpus Archive: $ARCHIVE_PATH"
echo "============================================================"

# Ensure target directories exist
mkdir -p hindi/data nepali/data

# Extract maintaining internal directory hierarchy
tar -xzvf "$ARCHIVE_PATH"

echo "============================================================"
echo "Corpora successfully unpacked to hindi/data/ and nepali/data/"
echo "============================================================"
