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
KERNELS="${KERNELS:-${KERNEL:-pqk fqk}}"
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

echo "GA rerun dataset: $DATASET"
echo "Kernels: $KERNELS"
echo "Config: qubits=$QUBITS depth=${DEPTH:-10*n} num_circuit=$NUM_CIRCUIT generations=$NUM_GENERATION p=$PROB_MUTATE train=$TRAINING_SIZE test=$TEST_SIZE"
echo "Holdout: run=$RUN_HOLDOUT qubits=$HOLDOUT_QUBITS seeds=$HOLDOUT_SEEDS test_size=$HOLDOUT_TEST_SIZE preprocess=$HOLDOUT_PREPROCESS"

run_kernel() {
  local kernel="$1"
  case "$kernel" in
    pqk|fqk) ;;
    *)
      echo "Unsupported kernel: $kernel" >&2
      exit 1
      ;;
  esac

  local run_dir="$REPO_ROOT/results/reviewer/ga_reruns/${DATASET}_${kernel}_n${QUBITS// /-}_${RUN_ID}"
  mkdir -p "$run_dir"

  echo "Output directory: $run_dir"
  echo "Running GA kernel=$kernel"

  cd "$run_dir"

  local train_args=(
    --data "$DATASET"
    --qubits $QUBITS
    --num-circuit "$NUM_CIRCUIT"
    --num-generation "$NUM_GENERATION"
    --prob-mutate "$PROB_MUTATE"
    --kernel "$kernel"
    --training-size "$TRAINING_SIZE"
    --test-size "$TEST_SIZE"
    --num-machines 1
    --id "$MACHINE_ID"
    --start-index "$START_INDEX"
  )

  if [[ -n "$DEPTH" ]]; then
    train_args+=(--depth $DEPTH)
  fi

  uv run python -m ga_qsvm.cli.train \
    "${train_args[@]}" \
    2>&1 | tee "$REPO_ROOT/logs/reviewer/ga_${DATASET}_${kernel}_n${QUBITS// /-}_${RUN_ID}.log"

  if [[ "$RUN_HOLDOUT" != "1" ]]; then
    echo "Done: $run_dir"
    return
  fi

  local artifact_dir
  artifact_dir="$(find "$run_dir" -maxdepth 1 -type d -name "${HOLDOUT_QUBITS}qubits_train_${kernel}_qsvm_*" | sort | tail -n 1)"
  if [[ -z "$artifact_dir" ]]; then
    echo "Could not find ${HOLDOUT_QUBITS}-qubit GA artifact under $run_dir" >&2
    exit 1
  fi

  local manifest="$run_dir/holdout_${DATASET}_${kernel}_n${HOLDOUT_QUBITS}_manifest.json"
  cat > "$manifest" <<JSON
{
  "description": "Fresh ${DATASET} GA-${kernel^^} n${HOLDOUT_QUBITS} circuit from ${RUN_ID} for repeated holdout.",
  "circuits": [
    {
      "id": "${DATASET}-ga-${kernel}-n${HOLDOUT_QUBITS}-${RUN_ID}",
      "dataset": "${DATASET}",
      "kernel": "ga-${kernel}",
      "path": "${artifact_dir}"
    }
  ]
}
JSON

  local holdout_models
  case "$kernel" in
    pqk)
      holdout_models="${HOLDOUT_MODELS:-rbf fixed-pqk ga-pqk}"
      ;;
    fqk)
      holdout_models="${HOLDOUT_MODELS:-fixed-fqk ga-fqk}"
      ;;
  esac

  local holdout_dir="$run_dir/holdout_${DATASET}_${kernel}_n${HOLDOUT_QUBITS}_${RUN_ID}"
  echo "Running repeated holdout benchmark"
  echo "Manifest: $manifest"
  echo "Artifact: $artifact_dir"
  echo "Models: $holdout_models"

  uv run python -m ga_qsvm.cli.frozen_benchmark \
    --manifest "$manifest" \
    --datasets "$DATASET" \
    --seeds $HOLDOUT_SEEDS \
    --test-size "$HOLDOUT_TEST_SIZE" \
    --preprocess "$HOLDOUT_PREPROCESS" \
    --models $holdout_models \
    --output-dir "$holdout_dir" \
    --n-features "$HOLDOUT_QUBITS" \
    --feature-dim-mode "$HOLDOUT_FEATURE_DIM_MODE" \
    --wandb-project "GA-QSVM-${DATASET}-${kernel}-holdout" \
    --wandb-name "holdout-${DATASET}-${kernel}-n${HOLDOUT_QUBITS}-${RUN_ID}" \
    --wandb-group "ga-${DATASET}-${kernel}-${RUN_ID}" \
    2>&1 | tee "$REPO_ROOT/logs/reviewer/holdout_${DATASET}_${kernel}_n${HOLDOUT_QUBITS}_${RUN_ID}.log"

  echo "Done: $run_dir"
}

for kernel in $KERNELS; do
  run_kernel "$kernel"
done
