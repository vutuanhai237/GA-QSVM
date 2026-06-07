#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Local-side reviewer benchmarks.
#
# Intended for the main workstation. It keeps the most important frozen Figure 6
# benchmark close to the existing artifacts, then optionally runs the slower
# random-FQK baseline. This script does not run noise and does not rerun GA.

export GA_QSVM_MAX_WORKERS="${GA_QSVM_MAX_WORKERS:-20}"
export DATASETS="${DATASETS:-digits wine cancer}"
export QUBITS="${QUBITS:-3 4 5 6 7}"
export SEEDS="${SEEDS:-100 101 102 103 104 105 106 107 108 109}"
export PREPROCESS="${PREPROCESS:-paper}"
export RANDOM_BUDGET="${RANDOM_BUDGET:-20}"

RUN_FROZEN_HOLDOUT="${RUN_FROZEN_HOLDOUT:-1}"
RUN_RANDOM_FQK="${RUN_RANDOM_FQK:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-0}"

# The checked-in manifest is currently the frozen q=7 Figure 6 circuit set.
# Override this if you later create a q=3..6 manifest.
export MANIFEST="${MANIFEST:-configs/reviewer/main_figure6_n7_manifest.json}"
export N_FEATURES="${N_FEATURES:-7}"
export FEATURE_DIM_MODE="${FEATURE_DIM_MODE:-global}"

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="logs/reviewer/local_all_${STAMP}"
mkdir -p "$LOG_DIR"

echo "Reviewer LOCAL benchmark schedule"
echo "  datasets:       $DATASETS"
echo "  qubits:         $QUBITS"
echo "  seeds:          $SEEDS"
echo "  max workers:    $GA_QSVM_MAX_WORKERS"
echo "  frozen manifest:$MANIFEST"
echo "  n features:     $N_FEATURES"
echo "  random budget:  $RANDOM_BUDGET"
echo "  log dir:        $LOG_DIR"

if [[ "$RUN_RANDOM_FQK" == "1" ]]; then
  echo
  echo "Step 1/3: random circuit search FQK baseline"
  (
    export KERNELS="fqk"
    export OUTPUT_ROOT="results/reviewer/random_search_baselines"
    "$SCRIPT_DIR/run_random_search_baseline_sweep.sh"
  ) 2>&1 | tee "$LOG_DIR/random_search_fqk.log"
else
  echo
  echo "Step 1/3: skipping random FQK because RUN_RANDOM_FQK=$RUN_RANDOM_FQK"
fi

if [[ "$RUN_FROZEN_HOLDOUT" == "1" ]]; then
  echo
  echo "Step 2/3: frozen Figure 6 repeated-holdout benchmark"
  (
    uv run python -m ga_qsvm.cli.frozen_benchmark \
      --manifest "$MANIFEST" \
      --datasets $DATASETS \
      --seeds $SEEDS \
      --test-size 0.3 \
      --preprocess "$PREPROCESS" \
      --models rbf fixed-pqk ga-pqk fixed-fqk ga-fqk \
      --output-dir results/reviewer/frozen_holdout_local_all \
      --n-features "$N_FEATURES" \
      --feature-dim-mode "$FEATURE_DIM_MODE"
  ) 2>&1 | tee "$LOG_DIR/frozen_holdout.log"
else
  echo
  echo "Step 2/3: skipping frozen holdout because RUN_FROZEN_HOLDOUT=$RUN_FROZEN_HOLDOUT"
fi

if [[ "$RUN_SUMMARY" == "1" ]]; then
  echo
  echo "Step 3/3: summarize reviewer results"
  "$SCRIPT_DIR/local_summarize_results.sh" 2>&1 | tee "$LOG_DIR/summary.log"
else
  echo
  echo "Step 3/3: skipping summary because RUN_SUMMARY=$RUN_SUMMARY"
fi

echo
echo "Local jobs complete. Main outputs:"
echo "  results/reviewer/frozen_holdout_local_all/"
echo "  results/reviewer/random_search_baselines/"
echo "  $LOG_DIR/"
