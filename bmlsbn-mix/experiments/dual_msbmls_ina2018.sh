#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-1}"
RUN_MODE="${RUN_MODE:-2day}"
MS_INCLUDE_SINGLETONS="${MS_INCLUDE_SINGLETONS:-1}"
MIXUP_LAMBDA_SAMPLING="${MIXUP_LAMBDA_SAMPLING:-epoch_antithetic}"
MS_LOSS_MAX_WEIGHT="${MS_LOSS_MAX_WEIGHT:-0.5}"
MS_LOSS_WARMUP_EPOCHS="${MS_LOSS_WARMUP_EPOCHS:-2}"
MS_LOSS_RAMP_EPOCHS="${MS_LOSS_RAMP_EPOCHS:-3}"
MS_NORMALIZE_PROTOTYPES="${MS_NORMALIZE_PROTOTYPES:-1}"
MS_LOGIT_SCALE="${MS_LOGIT_SCALE:-30.0}"

# Four workers are a balanced starting point for four GPUs. Keep prefetching
# conservative because each mixed-label training sample opens multiple files.
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
        CONFIG="./config/iNaturalist18/dual_msbmls_iNaturalist18_2day.py"
        WARMUP_EPOCH=1
        EXP_STR="dual_msbmls_iNaturalist18_4gpu_2day"
        ;;
    36h)
        CONFIG="./config/iNaturalist18/dual_msbmls_iNaturalist18_36h.py"
        WARMUP_EPOCH=1
        EXP_STR="dual_msbmls_iNaturalist18_4gpu_36h"
        ;;
    full)
        CONFIG="./config/iNaturalist18/dual_msbmls_iNaturalist18.py"
        WARMUP_EPOCH=5
        EXP_STR="dual_msbmls_iNaturalist18_full"
        ;;
    *)
        echo "RUN_MODE must be '36h', '2day', or 'full'." >&2
        exit 2
        ;;
esac

if [[ "${MS_INCLUDE_SINGLETONS}" == "1" ]]; then
    EXP_STR="${EXP_STR}_ms_global_singletons"
elif [[ "${MS_INCLUDE_SINGLETONS}" != "0" ]]; then
    echo "MS_INCLUDE_SINGLETONS must be 0 or 1." >&2
    exit 2
fi

if [[ "${MS_NORMALIZE_PROTOTYPES}" == "1" ]]; then
    EXP_STR="${EXP_STR}_msnorm_s${MS_LOGIT_SCALE}"
elif [[ "${MS_NORMALIZE_PROTOTYPES}" != "0" ]]; then
    echo "MS_NORMALIZE_PROTOTYPES must be 0 or 1." >&2
    exit 2
fi

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

EXP_STR="${EXP_STR}_msw${MS_LOSS_MAX_WEIGHT}_mswarm${MS_LOSS_WARMUP_EPOCHS}_ramp${MS_LOSS_RAMP_EPOCHS}"

ARGS=(
    --config "${CONFIG}"
    --loss-type dual_mixup
    --mixup-beta 0.2
    --mixup-lambda-sampling "${MIXUP_LAMBDA_SAMPLING}"
    --ms-loss-max-weight "${MS_LOSS_MAX_WEIGHT}"
    --ms-loss-warmup-epochs "${MS_LOSS_WARMUP_EPOCHS}"
    --ms-loss-ramp-epochs "${MS_LOSS_RAMP_EPOCHS}"
    --ms-logit-scale "${MS_LOGIT_SCALE}"
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
if [[ "${MS_INCLUDE_SINGLETONS}" == "1" ]]; then
    ARGS+=(--ms-include-unmixed-singletons)
fi
if [[ "${MS_NORMALIZE_PROTOTYPES}" == "1" ]]; then
    ARGS+=(--ms-normalize-prototypes)
fi

python3 -u dual_msbmls_main.py "${ARGS[@]}"
