#!/usr/bin/env bash
set -euo pipefail

# Script to package all heavy dataset files into a single archive
# Usage: ./scripts/zip_corpora.sh [output_zip_path]

OUTPUT_ZIP="${1:-corpora_dataset_bundle.tar.gz}"

echo "============================================================"
echo "Packaging LMA Heavy Corpus Files into: $OUTPUT_ZIP"
echo "============================================================"

# List of heavy files to bundle
FILES_TO_PACK=(
    "hindi/data/train.jsonl"
    "hindi/data/val.jsonl"
    "hindi/data/test.jsonl"
    "hindi/data/scraped_corpus.jsonl"
    "hindi/data/public_corpus.jsonl"
    "nepali/data/train.jsonl"
    "nepali/data/val.jsonl"
    "nepali/data/test.jsonl"
    "nepali/data/scraped_corpus.jsonl"
    "nepali/data/public_corpus.jsonl"
)

EXISTING_FILES=()
for f in "${FILES_TO_PACK[@]}"; do
    if [ -f "$f" ]; then
        EXISTING_FILES+=("$f")
        echo " [+] Found: $f ($(du -h "$f" | cut -f1))"
    else
        echo " [-] Skipping (not found): $f"
    fi
done

if [ ${#EXISTING_FILES[@]} -eq 0 ]; then
    echo "Error: No corpus files found to archive."
    exit 1
fi

echo "Creating compressed tarball..."
tar -czvf "$OUTPUT_ZIP" "${EXISTING_FILES[@]}"

echo "============================================================"
echo "Successfully created: $OUTPUT_ZIP ($(du -h "$OUTPUT_ZIP" | cut -f1))"
echo "============================================================"
