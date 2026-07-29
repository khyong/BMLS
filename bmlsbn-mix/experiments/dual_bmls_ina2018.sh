#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-1}"
RUN_MODE="${RUN_MODE:-2day}"
MIXUP_LAMBDA_SAMPLING="${MIXUP_LAMBDA_SAMPLING:-epoch_antithetic}"
RB_LOSS_MAX_WEIGHT="${RB_LOSS_MAX_WEIGHT:-0.5}"
RB_LOSS_WARMUP_EPOCHS="${RB_LOSS_WARMUP_EPOCHS:-2}"
RB_LOSS_RAMP_EPOCHS="${RB_LOSS_RAMP_EPOCHS:-3}"

export BMLSBN_NUM_WORKERS="${BMLSBN_NUM_WORKERS:-4}"
export BMLSBN_PREFETCH_FACTOR="${BMLSBN_PREFETCH_FACTOR:-1}"
export BMLSBN_SINGLE_TRAIN_VIEW="${BMLSBN_SINGLE_TRAIN_VIEW:-1}"
export BMLSBN_INAT_ROOT="${BMLSBN_INAT_ROOT:-/home/hoyong/datasets/ina2018_short256}"
export BMLSBN_INAT_LIST_ROOT="${BMLSBN_INAT_LIST_ROOT:-/home/hoyong/datasets/data_txt}"

if [[ ! -d "${BMLSBN_INAT_ROOT}/images" ]]; then
    echo "Aspect-preserving iNaturalist images were not found at ${BMLSBN_INAT_ROOT}/images." >&2
    echo "Run data/resize_ina2018.py before starting this experiment." >&2
    exit 2
fi

case "${RUN_MODE}" in
    2day)
        CONFIG="./config/iNaturalist18/dual_bmls_iNaturalist18_2day.py"
        WARMUP_EPOCH=1
        EXP_STR="dual_bmls_linear_iNaturalist18_4gpu_2day"
        ;;
    36h)
        CONFIG="./config/iNaturalist18/dual_bmls_iNaturalist18_36h.py"
        WARMUP_EPOCH=1
        EXP_STR="dual_bmls_linear_iNaturalist18_4gpu_36h"
        ;;
    full)
        CONFIG="./config/iNaturalist18/dual_bmls_iNaturalist18.py"
        WARMUP_EPOCH=5
        EXP_STR="dual_bmls_linear_iNaturalist18_full"
        ;;
    *)
        echo "RUN_MODE must be '36h', '2day', or 'full'." >&2
        exit 2
        ;;
esac

case "${MIXUP_LAMBDA_SAMPLING}" in
    epoch_antithetic)
        EXP_STR="${EXP_STR}_antithetic"
        ;;
    epoch|batch)
        ;;
    *)
        echo "MIXUP_LAMBDA_SAMPLING must be 'epoch_antithetic', 'epoch', or 'batch'." >&2
        exit 2
        ;;
esac

EXP_STR="${EXP_STR}_rbw${RB_LOSS_MAX_WEIGHT}_warm${RB_LOSS_WARMUP_EPOCHS}_ramp${RB_LOSS_RAMP_EPOCHS}"

ARGS=(
    --config "${CONFIG}"
    --loss-type dual_mixup
    --mixup-beta 0.2
    --mixup-lambda-sampling "${MIXUP_LAMBDA_SAMPLING}"
    --rb-classifier linear
    --ms-loss-max-weight "${RB_LOSS_MAX_WEIGHT}"
    --ms-loss-warmup-epochs "${RB_LOSS_WARMUP_EPOCHS}"
    --ms-loss-ramp-epochs "${RB_LOSS_RAMP_EPOCHS}"
    --train-rule DualBMLS
    --sampler-type default
    --weight-gamma inf
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

python3 -u dual_msbmls_main.py "${ARGS[@]}"
