#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROOTS=${ROOTS:-"results/reviewer/ga_reruns logs/reviewer"}
OUTPUT_DIR=${OUTPUT_DIR:-"results/reviewer/early_stop_analysis"}
PATIENCE=${PATIENCE:-50}
CHECKPOINTS=${CHECKPOINTS:-"50 100 150 200"}
TOLERANCE=${TOLERANCE:-0.0}
MIN_DELTA=${MIN_DELTA:-0.0}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache}

mkdir -p "$OUTPUT_DIR"

if [[ "$#" -gt 0 ]]; then
  ROOT_ARGS=("$@")
else
  # shellcheck disable=SC2206
  ROOT_ARGS=($ROOTS)
fi

uv run python -m ga_qsvm.cli.early_stop_analysis \
  --roots "${ROOT_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --patience "$PATIENCE" \
  --checkpoints $CHECKPOINTS \
  --tolerance "$TOLERANCE" \
  --min-delta "$MIN_DELTA"

echo "CSV: $OUTPUT_DIR/early_stop_analysis.csv"
echo "Markdown: $OUTPUT_DIR/early_stop_analysis.md"
