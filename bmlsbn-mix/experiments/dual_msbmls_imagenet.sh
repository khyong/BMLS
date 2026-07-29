#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-1}"
SCHEDULE="${SCHEDULE:-standard}"

case "${SCHEDULE}" in
    standard)
        CONFIG="./config/imagenetLT/dual_msbmls_imagenetLT.py"
        ;;
    extended)
        CONFIG="./config/imagenetLT/dual_msbmls_imagenetLT_extended.py"
        ;;
    *)
        echo "SCHEDULE must be 'standard' or 'extended'." >&2
        exit 2
        ;;
esac

ARGS=(
    --config "${CONFIG}"
    --loss-type dual_mixup
    --mixup-beta 0.2
    --train-rule DualBMLS
    --sampler-type default
    --weight-gamma inf
    --augmentation default
    --temp-eta 9
    --temp-epsilon 0.2
    --use-experts
    --use-experts-verbose
    --exp-str "dual_msbmls_imagenetLT_${SCHEDULE}"
    --seed "${SEED}"
)

python3 -u dual_msbmls_main.py "${ARGS[@]}"
