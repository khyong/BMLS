#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-1}"
RUN_MODE="${RUN_MODE:-2day}"

# Four workers are a balanced starting point for four GPUs. Keep prefetching
# conservative because each mixed-label training sample opens multiple files.
export BMLSBN_NUM_WORKERS="${BMLSBN_NUM_WORKERS:-4}"
export BMLSBN_PREFETCH_FACTOR="${BMLSBN_PREFETCH_FACTOR:-1}"
export BMLSBN_SINGLE_TRAIN_VIEW="${BMLSBN_SINGLE_TRAIN_VIEW:-1}"
export BMLSBN_INAT_ROOT="${BMLSBN_INAT_ROOT:-/data/hoyong/ina2018_short256}"
export BMLSBN_INAT_LIST_ROOT="${BMLSBN_INAT_LIST_ROOT:-/data/hoyong/data_txt}"

if [[ ! -d "${BMLSBN_INAT_ROOT}/images" ]]; then
    echo "Aspect-preserving iNaturalist images were not found at ${BMLSBN_INAT_ROOT}/images." >&2
    echo "Run data/resize_ina2018.py before starting this experiment." >&2
    exit 2
fi

case "${RUN_MODE}" in
    2day)
        CONFIG="./config/iNaturalist18/iNaturalist18_2day.py"
        WARMUP_EPOCH=1
        EXP_STR="iNaturalist18_4gpu_2day"
        ;;
    36h)
        CONFIG="./config/iNaturalist18/iNaturalist18_36h.py"
        WARMUP_EPOCH=1
        EXP_STR="iNaturalist18_4gpu_36h"
        ;;
    full)
        CONFIG="./config/iNaturalist18/iNaturalist18.py"
        WARMUP_EPOCH=5
        EXP_STR="iNaturalist18_full"
        ;;
    *)
        echo "RUN_MODE must be '36h', '2day', or 'full'." >&2
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
    --temp-eta 10
    --temp-epsilon 0.02
    --use-experts
    --use-experts-verbose
    --warmup-epoch "${WARMUP_EPOCH}"
    --exp-str "${EXP_STR}"
    --seed "${SEED}"
)

if [[ -n "${RESUME:-}" ]]; then
    ARGS+=(--resume "${RESUME}")
fi

python3 -u main.py "${ARGS[@]}"
