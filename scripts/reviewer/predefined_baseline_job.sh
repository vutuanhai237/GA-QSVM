#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "Usage: $0 <dataset> <ansatz> <kernel> <qubits>" >&2
  echo "Example: SEEDS='100 101' $0 wine efficient-su2 fqk 7" >&2
  exit 2
fi

DATASET="$1"
ANSATZ="$2"
KERNEL="$3"
QUBITS="$4"
SEEDS="${SEEDS:-100 101 102 103 104 105 106 107 108 109}"
PREPROCESS="${PREPROCESS:-paper}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/reviewer/predefined_baselines}"
MODEL="${ANSATZ}-${KERNEL}"
SEED_TAG="$(echo "$SEEDS" | tr ' ' '-')"
OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}_${MODEL}_q${QUBITS}_seeds-${SEED_TAG}"

echo "Running predefined baseline job"
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Qubits: $QUBITS"
echo "Seeds: $SEEDS"
echo "Output: $OUTPUT_DIR"

uv run python -m ga_qsvm.cli.predefined_benchmark \
  --datasets "$DATASET" \
  --qubits "$QUBITS" \
  --seeds $SEEDS \
  --test-size 0.3 \
  --preprocess "$PREPROCESS" \
  --models "$MODEL" \
  --output-dir "$OUTPUT_DIR"
