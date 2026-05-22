#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Round 2 cloud job for a single available Google Cloud VM.
#
# Run this script once on the cloud VM after the current Digits job finishes.
# It covers the remaining cloud-side work for the paper data:
#   1. Digits FQK, in case the current cloud run only covered PQK.
#   2. Cancer PQK and FQK.
#
# Local machine should cover Wine with the companion local round2 script.

export QUBITS="${QUBITS:-3 4 5 6 7}"
export NUM_CIRCUIT="${NUM_CIRCUIT:-20}"
export NUM_GENERATION="${NUM_GENERATION:-200}"
export PROB_MUTATE="${PROB_MUTATE:-0.1}"
export RUN_HOLDOUT="${RUN_HOLDOUT:-1}"
export HOLDOUT_QUBITS="${HOLDOUT_QUBITS:-7}"

echo "Round 2 CLOUD schedule"
echo "QUBITS=$QUBITS NUM_CIRCUIT=$NUM_CIRCUIT NUM_GENERATION=$NUM_GENERATION"

echo
echo "Step 1/2: Digits FQK only"
KERNELS=fqk "$SCRIPT_DIR/cloud_ga_dataset.sh" digits 0

echo
echo "Step 2/2: Cancer PQK + FQK"
KERNELS="pqk fqk" "$SCRIPT_DIR/cloud_ga_dataset.sh" cancer 0

echo
echo "Cloud round2 complete."
