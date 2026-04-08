#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/lab_graphicon_server.env}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 MODEL_NAME [extra args for tools/sweep_model.py]" >&2
  exit 1
fi

MODEL_NAME="$1"
shift

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

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

python "${REPO_ROOT}/tools/sweep_model.py" \
  --repo-root "${REPO_ROOT}" \
  --model "${MODEL_NAME}" \
  --device "${DEVICE:-auto}" \
  --gaze-csv-dir "${GAZE_CSV_DIR}" \
  --mesh-dir "${MESH_DIR}" \
  --json-dir "${JSON_DIR}" \
  --gt-dir "${GT_DIR}" \
  --output-root "${REPO_ROOT}/sweeps/${MODEL_NAME}" \
  "$@"
