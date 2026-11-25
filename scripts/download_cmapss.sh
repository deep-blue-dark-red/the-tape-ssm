#!/bin/bash
# Download NASA C-MAPSS Turbofan Engine Degradation Simulation dataset.
#
# If the automatic download fails, manually download from:
#   https://data.nasa.gov/download/xaut-bemq/application%2Fzip
#   or: https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
#
# Then unzip into data/cmapss/
set -e

DIR="$(cd "$(dirname "$0")/.." && pwd)/data/cmapss"
mkdir -p "$DIR"

# Check if already downloaded
if [ -f "$DIR/train_FD001.txt" ]; then
    echo "C-MAPSS data already exists in $DIR"
    ls -lh "$DIR"/*.txt
    exit 0
fi

echo "=== Downloading C-MAPSS dataset ==="

# PHM Society mirror of the NASA PCoE archive (works as of 2026-11-25; the zip nests CMAPSSData.zip)
URL1="https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
# Older NASA links (404 as of 2026-11-25), kept as fallbacks
URL2="https://data.nasa.gov/download/xaut-bemq/application%2Fzip"
URL3="https://ti.arc.nasa.gov/c/6/"

ZIPFILE="$DIR/CMAPSSData.zip"

for URL in "$URL1" "$URL2" "$URL3"; do
    echo "  Trying: $URL"
    if curl -sL --fail -o "$ZIPFILE" "$URL" 2>/dev/null; then
        # Check if it's actually a zip file
        if file "$ZIPFILE" | grep -q "Zip archive"; then
            echo "  Downloaded successfully!"
            break
        fi
    fi
    rm -f "$ZIPFILE"
done

if [ ! -f "$ZIPFILE" ]; then
    echo ""
    echo "ERROR: Automatic download failed."
    echo ""
    echo "Please download manually:"
    echo "  1. Go to: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/"
    echo "  2. Download 'Turbofan Engine Degradation Simulation Data Set'"
    echo "  3. Or from Kaggle: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps"
    echo "  4. Unzip the contents into: $DIR"
    echo ""
    echo "Expected files: train_FD001.txt, test_FD001.txt, RUL_FD001.txt, etc."
    exit 1
fi

echo "  Extracting..."
cd "$DIR"
unzip -o "$ZIPFILE"
rm -f "$ZIPFILE"

# The PHM mirror wraps the NASA archive: "6. Turbofan .../CMAPSSData.zip"
INNER=$(find . -name CMAPSSData.zip -not -path ./CMAPSSData.zip | head -1)
if [ -n "$INNER" ]; then
    unzip -o "$INNER"
    rm -rf "./6. Turbofan Engine Degradation Simulation Data Set"
fi

# Handle nested directory if present
if [ -d "CMAPSSData" ]; then
    mv CMAPSSData/* . 2>/dev/null || true
    rmdir CMAPSSData 2>/dev/null || true
fi

echo ""
echo "=== Done ==="
if [ -f "$DIR/../checksums/cmapss.sha256" ]; then
    (cd "$DIR/../.." && grep -v nasa-cmaps.zip data/checksums/cmapss.sha256 | shasum -a 256 -c -) || echo "Warning: checksum mismatch"
fi
ls -lh "$DIR"/*.txt 2>/dev/null || echo "Warning: no .txt files found"
