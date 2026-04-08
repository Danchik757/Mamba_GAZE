#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="/Users/admin/Documents/LAB/SALIENCY_code/GAZE_DATA"

MODEL="${1:-Aquarium_Deep_Sea_Diver_v1_L1}"
TARGET_DIR="${2:-/tmp/test1}"
ARCHIVE_PATH="${3:-/tmp/${MODEL}_test1_bundle.tar.gz}"

CSV_PATH="${DATA_ROOT}/csv_for_models/MeshMamba_non_texture/${MODEL}.csv"
JSON_PATH="${DATA_ROOT}/jsons_for_models/Mamba_non_textured/MeshMamba_non_texture_${MODEL}.json"
MESH_DIR="${DATA_ROOT}/datasets/MeshMamba/MeshMambaSaliency/MeshFile/non_texture"
GT_PATH="${DATA_ROOT}/datasets/MeshMamba/MeshMambaSaliency/SaliencyMap/non_texture/${MODEL}.csv"

for required_path in "${CSV_PATH}" "${JSON_PATH}" "${MESH_DIR}/${MODEL}" "${GT_PATH}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required path not found: ${required_path}" >&2
    exit 1
  fi
done

python "${REPO_ROOT}/tools/build_self_contained_workspace.py" \
  --repo-root "${REPO_ROOT}" \
  --target-dir "${TARGET_DIR}" \
  --model "${MODEL}" \
  --csv-path "${CSV_PATH}" \
  --json-path "${JSON_PATH}" \
  --mesh-dir "${MESH_DIR}" \
  --gt-path "${GT_PATH}"

tar -C "$(dirname "${TARGET_DIR}")" -czf "${ARCHIVE_PATH}" "$(basename "${TARGET_DIR}")"

echo "Bundle ready:"
echo "  workspace: ${TARGET_DIR}"
echo "  archive:   ${ARCHIVE_PATH}"
echo
echo "Copy to ubunt:"
echo "  scp ${ARCHIVE_PATH} ubu@<HOST>:/home/ubu/Documents/GAZE/"
echo
echo "On ubunt:"
echo "  mkdir -p /home/ubu/Documents/GAZE"
echo "  tar -xzf /home/ubu/Documents/GAZE/$(basename "${ARCHIVE_PATH}") -C /home/ubu/Documents/GAZE"
echo "  mv /home/ubu/Documents/GAZE/$(basename "${TARGET_DIR}") /home/ubu/Documents/GAZE/test1"
echo "  cd /home/ubu/Documents/GAZE/test1"
echo "  CONFIG_PATH=configs/test1_local.env bash scripts/create_conda_env.sh"
echo "  CONFIG_PATH=configs/test1_local.env bash scripts/run_model_server.sh ${MODEL} --device cpu --precompute-all-frames"
