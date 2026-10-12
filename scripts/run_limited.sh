#!/usr/bin/env bash
# Game-friendly training wrapper: cap CPU so desktop/gaming stays smooth.
# Recipe from tabular-training guidance: 2-thread libs + 200% CPU quota
# (2 core-equivalents) + low priority. Tune QUOTA if game stutters (100%)
# or training is too slow and game is fine (300%).
#
# Usage: scripts/run_limited.sh <quota%> <command...>
#   scripts/run_limited.sh 200 ~/.venvs/tabular/bin/python scripts/train_baseline.py ...
set -u
QUOTA="${1:-200}"
shift
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export LGBM_THREADS=2 XGB_THREADS=2
exec systemd-run --user --scope --collect \
  -p CPUQuota="${QUOTA}%" \
  -p CPUWeight=20 \
  nice -n 15 ionice -c 3 "$@"
