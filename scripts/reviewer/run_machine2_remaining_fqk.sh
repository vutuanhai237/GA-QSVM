#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Machine-2 queue for the remaining predefined FQK reviewer baselines.
#
# This is intentionally serial: one heavy FQK job at a time. Each job may still
# use the repository's internal worker pool, capped by GA_QSVM_MAX_WORKERS.

export GA_QSVM_MAX_WORKERS="${GA_QSVM_MAX_WORKERS:-20}"
export SEEDS="${SEEDS:-100 101 102 103 104 105 106 107 108 109}"
export PREPROCESS="${PREPROCESS:-paper}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-results/reviewer/predefined_baselines}"

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="logs/reviewer/machine2_remaining_fqk_${STAMP}"
mkdir -p "$LOG_DIR"

echo "Machine-2 remaining FQK queue"
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

# Bottom-up mirror of the remaining queue. Do not include
# cancer efficient-su2 FQK q7; the main machine is already running it.
run_job cancer two-local fqk 7
run_job cancer two-local fqk 6
run_job cancer two-local fqk 5
run_job cancer two-local fqk 4
run_job cancer two-local fqk 3

run_job digits two-local fqk 7
run_job digits two-local fqk 6
run_job digits two-local fqk 5
run_job digits two-local fqk 4
run_job digits two-local fqk 3

run_job digits efficient-su2 fqk 7

echo "Machine-2 queue complete."
echo "Copy these back to the main machine after completion:"
echo "  ${OUTPUT_ROOT}/"
echo "  ${LOG_DIR}/"
