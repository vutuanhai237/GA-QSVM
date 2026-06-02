#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/qsvm/refactor/GA-QSVM/.venv/bin/python}"
NOISES="${NOISES:-0.1 0.2 0.3 0.4 0.5}"
SHOTS="${SHOTS:-10000}"
WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_MODE
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

"$PYTHON" benchmark.py \
  --data digits \
  --kernel pqk \
  --qubits 5 \
  --num_cnot 14 \
  --depth 10 \
  --num-circuit 16 \
  --num-generation 200 \
  --prob-mutate 0.1 \
  --readout-noise $NOISES \
  --shots "$SHOTS"
