#!/usr/bin/env bash
set -euo pipefail

# Use the server's Python interpreter and pass the same YAML through every stage.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "${SCRIPT_DIR}/.."
INSPECTION_PYTHON=${INSPECTION_PYTHON:-python}
INSPECTION_CONFIG=${INSPECTION_CONFIG:-configs/inspection.yaml}
INSPECTED_BOARD_PATH=${1:-data/bad_example/320XLF4CQH00445.npy}

exec "${INSPECTION_PYTHON}" -m scripts.inspect \
    --config-path "${INSPECTION_CONFIG}" \
    --new-hexaboard-path "${INSPECTED_BOARD_PATH}" \
    --force
