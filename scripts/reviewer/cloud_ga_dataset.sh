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
NUM_CIRCUIT="${NUM_CIRCUIT:-100}"
NUM_GENERATION="${NUM_GENERATION:-200}"
PROB_MUTATE="${PROB_MUTATE:-0.1}"
KERNEL="${KERNEL:-pqk}"
TRAINING_SIZE="${TRAINING_SIZE:-100}"

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
echo "Note: this runs the GA search. It does not run final benchmark seeds 100-109."

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

echo "Done: $RUN_DIR"
