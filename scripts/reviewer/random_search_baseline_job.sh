#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 <dataset> <kernel> <qubits>" >&2
  echo "Example: SEEDS='100 101' RANDOM_BUDGET=20 $0 digits pqk 7" >&2
  exit 2
fi

DATASET="$1"
KERNEL="$2"
QUBITS="$3"
SEEDS="${SEEDS:-100 101 102 103 104 105 106 107 108 109}"
RANDOM_BUDGET="${RANDOM_BUDGET:-20}"
PREPROCESS="${PREPROCESS:-paper}"
DEPTH_MULTIPLIER="${DEPTH_MULTIPLIER:-5}"
NUM_CNOT_MULTIPLIER="${NUM_CNOT_MULTIPLIER:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/reviewer/random_search_baselines}"
MODEL="random-${KERNEL}"
SEED_TAG="$(echo "$SEEDS" | tr ' ' '-')"
OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}_${MODEL}_q${QUBITS}_budget${RANDOM_BUDGET}_seeds-${SEED_TAG}"

echo "Running random-search baseline job"
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Qubits: $QUBITS"
echo "Budget: $RANDOM_BUDGET"
echo "Seeds: $SEEDS"
echo "Output: $OUTPUT_DIR"

uv run python -m ga_qsvm.cli.random_search_benchmark \
  --datasets "$DATASET" \
  --qubits "$QUBITS" \
  --seeds $SEEDS \
  --test-size 0.3 \
  --preprocess "$PREPROCESS" \
  --models "$MODEL" \
  --random-budget "$RANDOM_BUDGET" \
  --depth-multiplier "$DEPTH_MULTIPLIER" \
  --num-cnot-multiplier "$NUM_CNOT_MULTIPLIER" \
  --output-dir "$OUTPUT_DIR"
