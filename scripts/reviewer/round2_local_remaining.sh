#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Run this on the local machine. It covers local-side work for the paper data:
#   1. Wine FQK, in case the current local run only covered PQK.
#   2. Cheap provenance/audit jobs.
#   3. Summary after cloud result folders are copied back into this checkout.
#
# If Wine FQK is already complete, set RUN_WINE_FQK=0.
# If cloud results have not been copied back yet, set RUN_SUMMARY=0 and rerun
# this script later with RUN_WINE_FQK=0 RUN_SUMMARY=1.

export QUBITS="${QUBITS:-3 4 5 6 7}"
export NUM_CIRCUIT="${NUM_CIRCUIT:-20}"
export NUM_GENERATION="${NUM_GENERATION:-200}"
export PROB_MUTATE="${PROB_MUTATE:-0.1}"
export RUN_HOLDOUT="${RUN_HOLDOUT:-1}"
export HOLDOUT_QUBITS="${HOLDOUT_QUBITS:-7}"
RUN_WINE_FQK="${RUN_WINE_FQK:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"

echo "Round 2 LOCAL schedule"
echo "QUBITS=$QUBITS NUM_CIRCUIT=$NUM_CIRCUIT NUM_GENERATION=$NUM_GENERATION"

if [[ "$RUN_WINE_FQK" == "1" ]]; then
  echo
  echo "Step 1/3: Wine FQK only"
  KERNELS=fqk "$SCRIPT_DIR/cloud_ga_dataset.sh" wine 0
else
  echo
  echo "Step 1/3: Skipping Wine FQK because RUN_WINE_FQK=$RUN_WINE_FQK"
fi

echo "Running reviewer tooling tests"
uv run pytest tests/test_reviewer_tooling.py -q

echo "Auditing artifacts from known roots and fresh ga_reruns"
uv run python -m ga_qsvm.cli.audit_artifacts \
  --roots /home/qsvm/benchmark/GA-QSVM /home/qsvm/test/GA-QSVM /home/qsvm/GA-QSVM results/reviewer/ga_reruns \
  --output results/reviewer/artifact_audit.csv

if [[ "$RUN_SUMMARY" == "1" ]]; then
  echo
  echo "Step 3/3: Summarizing holdout result folders"
  "$SCRIPT_DIR/local_summarize_results.sh"
else
  echo
  echo "Step 3/3: Skipping summary because RUN_SUMMARY=$RUN_SUMMARY"
fi

echo "Done. Check:"
echo "  results/reviewer/artifact_audit.csv"
echo "  results/reviewer/final/reviewer_summary.csv"
