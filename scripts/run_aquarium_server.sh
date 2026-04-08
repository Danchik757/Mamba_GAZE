#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_model_server.sh" "Aquarium_Deep_Sea_Diver_v1_L1" "$@"
