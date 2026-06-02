#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS="${DATASETS:-digits wine cancer}"
ANSATZES="${ANSATZES:-efficient-su2 two-local}"
KERNELS="${KERNELS:-pqk fqk}"
QUBITS="${QUBITS:-3 4 5 6 7}"
SEEDS="${SEEDS:-100 101 102 103 104 105 106 107 108 109}"

export SEEDS

for dataset in $DATASETS; do
  for ansatz in $ANSATZES; do
    for kernel in $KERNELS; do
      for qubits in $QUBITS; do
        "$SCRIPT_DIR/predefined_baseline_job.sh" "$dataset" "$ansatz" "$kernel" "$qubits"
      done
    done
  done
done
