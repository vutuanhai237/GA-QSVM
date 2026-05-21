#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# These are final benchmark split seeds, not independent GA seeds.
# They keep the selected/frozen circuits fixed and repeat train/test splits.
SEEDS="${SEEDS:-100 101 102 103 104 105 106 107 108 109}"
MANIFEST="${MANIFEST:-configs/reviewer/main_figure6_n7_manifest.json}"
DATASETS="${DATASETS:-digits wine cancer}"
N_FEATURES="${N_FEATURES:-7}"
PREPROCESS="${PREPROCESS:-paper}"

echo "Running local repeated-holdout benchmark for cheap models"
echo "Manifest: $MANIFEST"
echo "Seeds: $SEEDS"
echo "Preprocess: $PREPROCESS"

uv run python -m ga_qsvm.cli.frozen_benchmark \
  --manifest "$MANIFEST" \
  --datasets $DATASETS \
  --seeds $SEEDS \
  --test-size 0.3 \
  --preprocess "$PREPROCESS" \
  --models rbf fixed-pqk ga-pqk \
  --output-dir results/reviewer/figure6_holdout_local_pqk_rbf \
  --n-features "$N_FEATURES"

cat <<'MSG'
Done: local cheap benchmark models.

FQK models are intentionally not run here by default because they are the slow path.
Run FQK on cloud VMs with scripts/reviewer/cloud_fqk_benchmark_dataset.sh.
MSG

