#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PILOT_CONFIG="./config/iNaturalist18/dual_msbmls_iNaturalist18.py"
export PILOT_TRAIN_RULE="DualBMLS"
export PILOT_DUAL_SAMPLE="false"
export PILOT_DUAL_SAMPLER="default"

exec "${SCRIPT_DIR}/pilot_ina2018.sh" "$@"
