#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

DATASET="${1:-}"
if [[ -z "$DATASET" ]]; then
  echo "Usage: $0 <digits|wine|cancer> [machine_id]" >&2
  exit 2
fi

case "$DATASET" in
  digits|wine|cancer) ;;
  *)
    echo "Unsupported dataset: $DATASET" >&2
    echo "Expected one of: digits, wine, cancer" >&2
    exit 2
    ;;
esac

MACHINE_ID="${2:-0}"

# GA search configuration for the revision rerun. One cloud VM owns one dataset
# and runs one legacy random-allocation GA search for each qubit count.
QUBITS="${QUBITS:-3 4 5 6 7}"
DEPTH="${DEPTH:-}"                # Empty means train CLI uses 10 * qubits.
NUM_CIRCUIT="${NUM_CIRCUIT:-20}"
NUM_GENERATION="${NUM_GENERATION:-200}"
PROB_MUTATE="${PROB_MUTATE:-0.1}"
KERNEL="${KERNEL:-pqk}"
TRAINING_SIZE="${TRAINING_SIZE:-100}"
HOLDOUT_QUBITS="${HOLDOUT_QUBITS:-7}"
HOLDOUT_SEEDS="${HOLDOUT_SEEDS:-100 101 102 103 104 105 106 107 108 109}"
HOLDOUT_TEST_SIZE="${HOLDOUT_TEST_SIZE:-0.3}"
HOLDOUT_PREPROCESS="${HOLDOUT_PREPROCESS:-paper}"
HOLDOUT_FEATURE_DIM_MODE="${HOLDOUT_FEATURE_DIM_MODE:-global}"
RUN_HOLDOUT="${RUN_HOLDOUT:-1}"

# Wine has only 178 samples, so 100 train + 100 test is impossible for GA search.
# The final reviewer benchmark uses repeated holdout separately.
TEST_SIZE="${TEST_SIZE:-50}"
START_INDEX="${START_INDEX:-0}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="$REPO_ROOT/results/reviewer/ga_reruns/${DATASET}_${KERNEL}_n${QUBITS// /-}_${RUN_ID}"
mkdir -p "$RUN_DIR"

echo "GA rerun dataset: $DATASET"
echo "Output directory: $RUN_DIR"
echo "Config: qubits=$QUBITS depth=${DEPTH:-10*n} num_circuit=$NUM_CIRCUIT generations=$NUM_GENERATION p=$PROB_MUTATE kernel=$KERNEL train=$TRAINING_SIZE test=$TEST_SIZE"
echo "Holdout: run=$RUN_HOLDOUT qubits=$HOLDOUT_QUBITS seeds=$HOLDOUT_SEEDS test_size=$HOLDOUT_TEST_SIZE preprocess=$HOLDOUT_PREPROCESS"

cd "$RUN_DIR"

TRAIN_ARGS=(
  --data "$DATASET"
  --qubits $QUBITS
  --num-circuit "$NUM_CIRCUIT"
  --num-generation "$NUM_GENERATION"
  --prob-mutate "$PROB_MUTATE"
  --kernel "$KERNEL"
  --training-size "$TRAINING_SIZE"
  --test-size "$TEST_SIZE"
  --num-machines 1
  --id "$MACHINE_ID"
  --start-index "$START_INDEX"
)

if [[ -n "$DEPTH" ]]; then
  TRAIN_ARGS+=(--depth $DEPTH)
fi

uv run python -m ga_qsvm.cli.train \
  "${TRAIN_ARGS[@]}" \
  2>&1 | tee "$REPO_ROOT/logs/reviewer/ga_${DATASET}_${KERNEL}_n${QUBITS// /-}_${RUN_ID}.log"

if [[ "$RUN_HOLDOUT" == "1" ]]; then
  ARTIFACT_DIR="$(find "$RUN_DIR" -maxdepth 1 -type d -name "${HOLDOUT_QUBITS}qubits_train_${KERNEL}_qsvm_*" | sort | tail -n 1)"
  if [[ -z "$ARTIFACT_DIR" ]]; then
    echo "Could not find ${HOLDOUT_QUBITS}-qubit GA artifact under $RUN_DIR" >&2
    exit 1
  fi

  MANIFEST="$RUN_DIR/holdout_${DATASET}_${KERNEL}_n${HOLDOUT_QUBITS}_manifest.json"
  cat > "$MANIFEST" <<JSON
{
  "description": "Fresh ${DATASET} GA-${KERNEL^^} n${HOLDOUT_QUBITS} circuit from ${RUN_ID} for repeated holdout.",
  "circuits": [
    {
      "id": "${DATASET}-ga-${KERNEL}-n${HOLDOUT_QUBITS}-${RUN_ID}",
      "dataset": "${DATASET}",
      "kernel": "ga-${KERNEL}",
      "path": "${ARTIFACT_DIR}"
    }
  ]
}
JSON

  case "$KERNEL" in
    pqk)
      HOLDOUT_MODELS="${HOLDOUT_MODELS:-rbf fixed-pqk ga-pqk}"
      ;;
    fqk)
      HOLDOUT_MODELS="${HOLDOUT_MODELS:-fixed-fqk ga-fqk}"
      ;;
    *)
      echo "Unsupported holdout kernel: $KERNEL" >&2
      exit 1
      ;;
  esac

  HOLDOUT_DIR="$RUN_DIR/holdout_${DATASET}_${KERNEL}_n${HOLDOUT_QUBITS}_${RUN_ID}"
  echo "Running repeated holdout benchmark"
  echo "Manifest: $MANIFEST"
  echo "Artifact: $ARTIFACT_DIR"
  echo "Models: $HOLDOUT_MODELS"

  uv run python -m ga_qsvm.cli.frozen_benchmark \
    --manifest "$MANIFEST" \
    --datasets "$DATASET" \
    --seeds $HOLDOUT_SEEDS \
    --test-size "$HOLDOUT_TEST_SIZE" \
    --preprocess "$HOLDOUT_PREPROCESS" \
    --models $HOLDOUT_MODELS \
    --output-dir "$HOLDOUT_DIR" \
    --n-features "$HOLDOUT_QUBITS" \
    --feature-dim-mode "$HOLDOUT_FEATURE_DIM_MODE" \
    --wandb-project "GA-QSVM-${DATASET}-${KERNEL}-holdout" \
    --wandb-name "holdout-${DATASET}-${KERNEL}-n${HOLDOUT_QUBITS}-${RUN_ID}" \
    --wandb-group "ga-${DATASET}-${KERNEL}-${RUN_ID}" \
    2>&1 | tee "$REPO_ROOT/logs/reviewer/holdout_${DATASET}_${KERNEL}_n${HOLDOUT_QUBITS}_${RUN_ID}.log"
fi

echo "Done: $RUN_DIR"
