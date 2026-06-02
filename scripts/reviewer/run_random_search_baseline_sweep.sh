#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS="${DATASETS:-digits wine cancer}"
KERNELS="${KERNELS:-pqk}"
QUBITS="${QUBITS:-7}"
SEEDS="${SEEDS:-100 101 102 103 104 105 106 107 108 109}"
RANDOM_BUDGET="${RANDOM_BUDGET:-20}"

export SEEDS RANDOM_BUDGET

for dataset in $DATASETS; do
  for kernel in $KERNELS; do
    for qubits in $QUBITS; do
      "$SCRIPT_DIR/random_search_baseline_job.sh" "$dataset" "$kernel" "$qubits"
    done
  done
done
