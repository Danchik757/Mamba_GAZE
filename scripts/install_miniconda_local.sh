#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${1:-${REPO_ROOT}/configs/test1_local.env}}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

INSTALL_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
INSTALLER_PATH="${INSTALLER_PATH:-/tmp/Miniconda3-latest-Linux-x86_64.sh}"
INSTALLER_URL="${INSTALLER_URL:-https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh}"

mkdir -p "$(dirname "${INSTALL_ROOT}")"

if [[ -x "${INSTALL_ROOT}/bin/conda" ]]; then
  echo "Miniconda already installed: ${INSTALL_ROOT}"
  exit 0
fi

echo "Downloading Miniconda installer..."
curl -L "${INSTALLER_URL}" -o "${INSTALLER_PATH}"

echo "Installing Miniconda to: ${INSTALL_ROOT}"
bash "${INSTALLER_PATH}" -b -p "${INSTALL_ROOT}"

# shellcheck disable=SC1091
source "${INSTALL_ROOT}/etc/profile.d/conda.sh"
conda --version

echo
echo "Miniconda ready: ${INSTALL_ROOT}"
echo "Next:"
echo "  CONFIG_PATH=${CONFIG_PATH} bash scripts/create_conda_env.sh"
