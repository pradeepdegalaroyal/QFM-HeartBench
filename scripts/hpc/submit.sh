#!/bin/bash
# Submit QFM-HeartBench worker jobs. Usage:
#   bash scripts/hpc/submit.sh <queue> <ncpus> <mem> <suites> <select> [workers]
# Examples:
#   bash scripts/hpc/submit.sh long  4 24gb  main small          # 4 single-thread workers
#   bash scripts/hpc/submit.sh long  8 64gb  main big 1          # 1 worker, 8 BLAS threads (CVD full)
#   bash scripts/hpc/submit.sh short 4 24gb  main,external,diag small
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PATH=/opt/pbs/bin:$PATH
Q=$1; N=$2; MEM=$3; SUITES=$4; SELECT=$5; WORKERS=${6:-$N}
mkdir -p "$ROOT/runs/pbs_logs"
qsub -q "$Q" -l "select=1:ncpus=$N:mem=$MEM" -l walltime=$([ "$Q" = short ] && echo 168:00:00 || echo 500:00:00) \
     -v "QFMB_ROOT=$ROOT,SUITES=$SUITES,SELECT=$SELECT,WORKERS=$WORKERS,QFMB_MEM=$MEM,WALLTIME_HOURS=$([ "$Q" = short ] && echo 168 || echo 500)" \
     -N "qfmb_${SELECT}" -o "$ROOT/runs/pbs_logs/" "$ROOT/scripts/hpc/qfmb_job.pbs"
