#!/bin/bash
# Creates the self-contained Python environment for QFM-HeartBench on the HPC.
# Usage: bash scripts/setup_env_hpc.sh   (run from the project root; about 10 min)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA=${CONDA:-conda}   # path to your conda executable
ENV="$ROOT/env"
if [ ! -x "$ENV/bin/python" ]; then
  "$CONDA" create -y -p "$ENV" --override-channels -c conda-forge \
    python=3.11 "numpy=1.26" "scipy=1.13" "scikit-learn=1.5" "pandas=2.2" \
    matplotlib statsmodels joblib psutil pyyaml tabulate xgboost lightgbm threadpoolctl
fi
"$ENV/bin/python" -m pip install --no-cache-dir "qiskit==2.2.3" "qiskit-aer==0.17.2"
"$ENV/bin/python" - <<'PY'
import importlib
for m in ["numpy","scipy","sklearn","pandas","qiskit","qiskit_aer","xgboost","lightgbm","statsmodels","matplotlib"]:
    print(m, importlib.import_module(m).__version__)
PY
echo ENV_READY
