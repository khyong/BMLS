#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-1}"
SCHEDULE="${SCHEDULE:-standard}"

case "${SCHEDULE}" in
    standard)
        CONFIG="./config/imagenetLT/imagenetLT.py"
        ;;
    extended)
        CONFIG="./config/imagenetLT/imagenetLT_extended.py"
        ;;
    *)
        echo "SCHEDULE must be 'standard' or 'extended'." >&2
        exit 2
        ;;
esac

ARGS=(
    --config "${CONFIG}"
    --loss-type mixup
    --mixup-beta 0.2
    --train-rule None
    --dual-sample True
    --sampler-type default
    --dual-sampler-type reverse
    --weight-gamma inf
    --use-dual-soft-weight
    --augmentation default
    --temp-eta 9
    --temp-epsilon 0.2
    --use-experts
    --use-experts-verbose
    --exp-str "imagenetLT_${SCHEDULE}"
    --seed "${SEED}"
)

python3 -u main.py "${ARGS[@]}"
