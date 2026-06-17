#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Run the currently missing EfficientSU2/TwoLocal predefined feature-map
# baselines for the reviewer response.
#
# This queue is intentionally serial because the missing jobs are FQK-heavy.
# Each job may still use the repository's internal circuit worker pool, capped by
# GA_QSVM_MAX_WORKERS. On a 20-vCPU VM, keep the default below unless RAM is tight.

export GA_QSVM_MAX_WORKERS="${GA_QSVM_MAX_WORKERS:-20}"
export SEEDS="${SEEDS:-100 101 102 103 104 105 106 107 108 109}"
export PREPROCESS="${PREPROCESS:-paper}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-results/reviewer/predefined_baselines}"

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="logs/reviewer/missing_predefined_feature_maps_${STAMP}"
mkdir -p "$LOG_DIR"

echo "Missing EfficientSU2/TwoLocal predefined feature-map queue"
echo "  jobs:        12 FQK jobs"
echo "  max workers: $GA_QSVM_MAX_WORKERS"
echo "  seeds:       $SEEDS"
echo "  preprocess:  $PREPROCESS"
echo "  output root: $OUTPUT_ROOT"
echo "  log dir:     $LOG_DIR"
echo

run_job() {
  local dataset="$1"
  local ansatz="$2"
  local kernel="$3"
  local qubits="$4"
  local model="${ansatz}-${kernel}"
  local seed_tag
  seed_tag="$(echo "$SEEDS" | tr ' ' '-')"
  local output_dir="${OUTPUT_ROOT}/${dataset}_${model}_q${qubits}_seeds-${seed_tag}"
  local log_file="${LOG_DIR}/${dataset}_${model}_q${qubits}.log"

  if [[ -f "${output_dir}/summary.csv" ]]; then
    echo "Skip existing: ${dataset} ${model} q=${qubits}"
    return 0
  fi

  echo "Run: ${dataset} ${model} q=${qubits}"
  "$SCRIPT_DIR/predefined_baseline_job.sh" "$dataset" "$ansatz" "$kernel" "$qubits" \
    2>&1 | tee "$log_file"
  echo
}

# Missing jobs from the current reviewer predefined-baseline sweep:
#   datasets: digits, breast cancer
#   ansatzes: EfficientSU2 q7 only; TwoLocal q3..q7
#   kernel:   FQK
#
# Order:
#   1. q3..q6 TwoLocal first, so the q=3..6 paper plot becomes complete early.
#   2. q7 jobs last, because they are the most memory-heavy.
run_job digits two-local fqk 3
run_job digits two-local fqk 4
run_job digits two-local fqk 5
run_job digits two-local fqk 6

run_job cancer two-local fqk 3
run_job cancer two-local fqk 4
run_job cancer two-local fqk 5
run_job cancer two-local fqk 6

run_job digits efficient-su2 fqk 7
run_job digits two-local fqk 7
run_job cancer efficient-su2 fqk 7
run_job cancer two-local fqk 7

echo "Missing predefined feature-map queue complete."
echo "Copy these directories back to the main machine after completion:"
echo "  ${OUTPUT_ROOT}/"
echo "  ${LOG_DIR}/"
