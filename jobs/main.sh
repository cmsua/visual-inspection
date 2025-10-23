#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   jobs/main.sh [inspected_hexaboard_path]
#
# Runs the inspection pipeline against the provided board (defaults to the bad example NPY).

INSPECTED_BOARD_PATH=${1:-./data/bad_example/aligned_images5.npy}
BASELINE_BOARD_PATH=./data/train/aligned_images1.npy

python -m scripts.inspect \
    -b "${BASELINE_BOARD_PATH}" \
    -n "${INSPECTED_BOARD_PATH}"
