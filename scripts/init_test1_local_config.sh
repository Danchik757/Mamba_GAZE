#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GAZE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
CONFIG_PATH="${1:-${REPO_ROOT}/configs/test1_local.env}"

mkdir -p "$(dirname "${CONFIG_PATH}")"

cat > "${CONFIG_PATH}" <<EOF
REPO_ROOT="${REPO_ROOT}"
ENV_ROOT="${GAZE_ROOT}/environments"
ENV_NAME="meshmamba_gaze"
ENV_PATH="\${ENV_ROOT}/\${ENV_NAME}"

RENDER_ROOT=""
VIDEOS_DIR=""
JSON_DIR="\${REPO_ROOT}/data/jsons_for_models/Mamba_non_textured"
MESH_DIR="\${REPO_ROOT}/data/datasets/MeshMamba/MeshMambaSaliency/MeshFile/non_texture"
GT_DIR="\${REPO_ROOT}/data/datasets/MeshMamba/MeshMambaSaliency/SaliencyMap/non_texture"
GAZE_CSV_DIR="\${REPO_ROOT}/data/csv_for_models/MeshMamba_non_texture"
OUTPUT_DIR="\${REPO_ROOT}/run_outputs"

TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
EOF

echo "Config written: ${CONFIG_PATH}"
