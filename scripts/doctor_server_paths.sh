#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/lab_graphicon_server.env}"
MODEL_NAME="${1:-Aquarium_Deep_Sea_Diver_v1_L1}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

echo "CONFIG_PATH=${CONFIG_PATH}"
echo "MODEL_NAME=${MODEL_NAME}"
echo

for var_name in REPO_ROOT ENV_PATH GAZE_CSV_DIR MESH_DIR JSON_DIR GT_DIR OUTPUT_DIR; do
  value="${!var_name}"
  echo "${var_name}=${value}"
done

echo
for target in "${GAZE_CSV_DIR}" "${MESH_DIR}" "${JSON_DIR}" "${GT_DIR}"; do
  echo "== ${target} =="
  if [[ -d "${target}" ]]; then
    ls -ld "${target}"
    find "${target}" -maxdepth 1 -type f | sed -n '1,5p'
    find "${target}" -maxdepth 1 -type d | sed -n '1,5p'
  else
    echo "MISSING_OR_NOT_DIR"
  fi
  echo
done

echo "== Aquarium lookup =="
find "${GAZE_CSV_DIR}" -maxdepth 1 -iname "*${MODEL_NAME}*" 2>/dev/null | sed -n '1,10p'
find "${MESH_DIR}" -maxdepth 2 -iname "*${MODEL_NAME}*" 2>/dev/null | sed -n '1,10p'
find "${JSON_DIR}" -maxdepth 1 -iname "*${MODEL_NAME}*" 2>/dev/null | sed -n '1,10p'
find "${GT_DIR}" -maxdepth 1 -iname "*${MODEL_NAME}*" 2>/dev/null | sed -n '1,10p'
