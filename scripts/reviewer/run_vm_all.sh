#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# VM-side reviewer benchmarks.
#
# Intended for a fresh 20-vCPU cloud VM. These jobs do not run GA. They evaluate
# fixed Qiskit ansatz baselines and the cheaper random-search PQK baseline using
# the same repeated-holdout data flow as the Figure 6 rerun.

export GA_QSVM_MAX_WORKERS="${GA_QSVM_MAX_WORKERS:-20}"
export DATASETS="${DATASETS:-digits wine cancer}"
export QUBITS="${QUBITS:-3 4 5 6 7}"
export SEEDS="${SEEDS:-100 101 102 103 104 105 106 107 108 109}"
export PREPROCESS="${PREPROCESS:-paper}"
export RANDOM_BUDGET="${RANDOM_BUDGET:-20}"

RUN_PREDEFINED="${RUN_PREDEFINED:-1}"
RUN_RANDOM_PQK="${RUN_RANDOM_PQK:-1}"

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="logs/reviewer/vm_all_${STAMP}"
mkdir -p "$LOG_DIR"

echo "Reviewer VM benchmark schedule"
echo "  datasets:       $DATASETS"
echo "  qubits:         $QUBITS"
echo "  seeds:          $SEEDS"
echo "  max workers:    $GA_QSVM_MAX_WORKERS"
echo "  random budget:  $RANDOM_BUDGET"
echo "  log dir:        $LOG_DIR"

if [[ "$RUN_PREDEFINED" == "1" ]]; then
  echo
  echo "Step 1/2: predefined Qiskit ansatz baselines"
  echo "Models: EfficientSU2-FQK/PQK and TwoLocal-FQK/PQK"
  (
    export ANSATZES="efficient-su2 two-local"
    export KERNELS="pqk fqk"
    export OUTPUT_ROOT="results/reviewer/predefined_baselines"
    "$SCRIPT_DIR/run_predefined_baseline_sweep.sh"
  ) 2>&1 | tee "$LOG_DIR/predefined_baselines.log"
else
  echo
  echo "Step 1/2: skipping predefined baselines because RUN_PREDEFINED=$RUN_PREDEFINED"
fi

if [[ "$RUN_RANDOM_PQK" == "1" ]]; then
  echo
  echo "Step 2/2: random circuit search PQK baseline"
  (
    export KERNELS="pqk"
    export OUTPUT_ROOT="results/reviewer/random_search_baselines"
    "$SCRIPT_DIR/run_random_search_baseline_sweep.sh"
  ) 2>&1 | tee "$LOG_DIR/random_search_pqk.log"
else
  echo
  echo "Step 2/2: skipping random PQK because RUN_RANDOM_PQK=$RUN_RANDOM_PQK"
fi

echo
echo "VM jobs complete. Copy these back to the main machine if this ran on cloud:"
echo "  results/reviewer/predefined_baselines/"
echo "  results/reviewer/random_search_baselines/"
echo "  $LOG_DIR/"
