#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

DATASET="${1:-}"
if [[ -z "$DATASET" ]]; then
  echo "Usage: $0 <digits|wine|cancer> [seed-list...]" >&2
  echo "Example: $0 digits 100 101 102 103 104" >&2
  exit 2
fi

case "$DATASET" in
  digits|wine|cancer) ;;
  *)
    echo "Unsupported dataset: $DATASET" >&2
    exit 2
    ;;
esac
shift || true

if [[ "$#" -gt 0 ]]; then
  SEEDS="$*"
else
  SEEDS="${SEEDS:-100 101 102 103 104 105 106 107 108 109}"
fi

MANIFEST="${MANIFEST:-configs/reviewer/main_figure6_n7_manifest.json}"
N_FEATURES="${N_FEATURES:-7}"
PREPROCESS="${PREPROCESS:-paper}"
FEATURE_DIM_MODE="${FEATURE_DIM_MODE:-global}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
SEED_TAG="$(echo "$SEEDS" | tr ' ' '-')"
OUT_DIR="results/reviewer/figure6_holdout_fqk_${DATASET}_${SEED_TAG}_${RUN_ID}"

echo "Running FQK repeated-holdout benchmark"
echo "Dataset: $DATASET"
echo "Seeds: $SEEDS"
echo "Output: $OUT_DIR"
echo "These seeds are final train/test split seeds, not GA seeds."

uv run python -m ga_qsvm.cli.frozen_benchmark \
  --manifest "$MANIFEST" \
  --datasets "$DATASET" \
  --seeds $SEEDS \
  --test-size 0.3 \
  --preprocess "$PREPROCESS" \
  --models fixed-fqk ga-fqk \
  --output-dir "$OUT_DIR" \
  --n-features "$N_FEATURES" \
  --feature-dim-mode "$FEATURE_DIM_MODE" \
  2>&1 | tee "logs/reviewer/fqk_${DATASET}_${SEED_TAG}_${RUN_ID}.log"

echo "Done: $OUT_DIR"

