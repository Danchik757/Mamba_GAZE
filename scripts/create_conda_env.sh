#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-${REPO_ROOT}/configs/lab_graphicon_server.env}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available in PATH" >&2
  exit 1
fi

mkdir -p "${ENV_ROOT}"

echo "Creating conda env at: ${ENV_PATH}"
conda env create -p "${ENV_PATH}" -f "${REPO_ROOT}/environment.server.yml" --force

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"

python -m pip install --upgrade pip
python -m pip install --index-url "${TORCH_INDEX_URL}" torch
python -m pip install -e "${REPO_ROOT}" --no-deps

echo "Environment ready: ${ENV_PATH}"
python - <<'PY'
import torch, numpy, pandas
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
PY
