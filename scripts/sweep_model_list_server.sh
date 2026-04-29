#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/lab_graphicon_server.env}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 MODEL_LIST_TXT [extra args for tools/sweep_model_list.py]" >&2
  exit 1
fi

MODEL_LIST_TXT="$1"
shift

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

if ! command -v conda >/dev/null 2>&1; then
  if [[ -n "${CONDA_ROOT:-}" && -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  fi
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

python "${REPO_ROOT}/tools/sweep_model_list.py" \
  --repo-root "${REPO_ROOT}" \
  --model-list "${MODEL_LIST_TXT}" \
  --output-root "${REPO_ROOT}/sweeps_multi" \
  --device "${DEVICE:-auto}" \
  --gaze-csv-dir "${GAZE_CSV_DIR}" \
  --mesh-dir "${MESH_DIR}" \
  --json-dir "${JSON_DIR}" \
  --gt-dir "${GT_DIR}" \
  "$@"
