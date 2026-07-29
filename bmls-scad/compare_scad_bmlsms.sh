#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/datasets}"
OUT_ROOT="${OUT_ROOT:-${REPO_DIR}/out/comparison}"
DATASETS="${DATASETS:-cifar10 cifar100 smallimagenet}"
SEEDS="${SEEDS:-0 1 2}"
ARCH="${ARCH:-wideresnet}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
TOTAL_STEPS="${TOTAL_STEPS:-250000}"
EVAL_STEP="${EVAL_STEP:-500}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LABELED_IMB_RATIO="${LABELED_IMB_RATIO:-100}"
UNLABELED_IMB_RATIO="${UNLABELED_IMB_RATIO:-100}"
SMALLIMAGENET_IMG_SIZES="${SMALLIMAGENET_IMG_SIZES:-${IMG_SIZE:-32 64}}"
BMLSMS_ALPHA="${BMLSMS_ALPHA:-1.0}"
BMLSMS_LAMBDA="${BMLSMS_LAMBDA:-1.0}"

run_experiment() {
    local dataset="$1"
    local method="$2"
    local seed="$3"
    local img_size="${4:-32}"
    local num_max
    local num_max_u

    case "${dataset}" in
        cifar10)
            num_max="${CIFAR10_NUM_MAX:-500}"
            num_max_u="${CIFAR10_NUM_MAX_U:-4000}"
            ;;
        cifar100)
            # Keep at least one labeled example in every fine class when the
            # default imbalance ratio is 100.
            num_max="${CIFAR100_NUM_MAX:-150}"
            num_max_u="${CIFAR100_NUM_MAX_U:-300}"
            ;;
        smallimagenet)
            # Small ImageNet currently uses its built-in 10% labeled split.
            num_max="${SMALLIMAGENET_NUM_MAX:-500}"
            num_max_u="${SMALLIMAGENET_NUM_MAX_U:-4000}"
            ;;
        *)
            echo "Unsupported dataset: ${dataset}" >&2
            return 2
            ;;
    esac

    local output_dir="${OUT_ROOT}/${dataset}/${method}/seed${seed}"
    if [[ "${dataset}" == "smallimagenet" ]]; then
        output_dir="${OUT_ROOT}/${dataset}/img${img_size}/${method}/seed${seed}"
    fi
    mkdir -p "${output_dir}"

    local command=(
        "${PYTHON_BIN}" -u acrwithSCAD.py
        --dataset "${dataset}"
        --data-root "${DATA_ROOT}"
        --arch "${ARCH}"
        --gpu-id "${GPU_ID}"
        --seed "${seed}"
        --num-max "${num_max}"
        --num-max-u "${num_max_u}"
        --imb-ratio-label "${LABELED_IMB_RATIO}"
        --imb-ratio-unlabel "${UNLABELED_IMB_RATIO}"
        --batch-size "${BATCH_SIZE}"
        --num-workers "${NUM_WORKERS}"
        --total-steps "${TOTAL_STEPS}"
        --eval-step "${EVAL_STEP}"
        --img-size "${img_size}"
        --superclass-method "${method}"
        --bmlsms-alpha "${BMLSMS_ALPHA}"
        --bmlsms-lambda "${BMLSMS_LAMBDA}"
        --out "${output_dir}"
    )

    echo "Running dataset=${dataset} img_size=${img_size} method=${method} seed=${seed}"
    "${command[@]}" 2>&1 | tee "${output_dir}/stdout.log"
}

for dataset in ${DATASETS}; do
    for seed in ${SEEDS}; do
        if [[ "${dataset}" == "smallimagenet" ]]; then
            for img_size in ${SMALLIMAGENET_IMG_SIZES}; do
                run_experiment "${dataset}" scad "${seed}" "${img_size}"
                run_experiment "${dataset}" bmlsms "${seed}" "${img_size}"
            done
        else
            run_experiment "${dataset}" scad "${seed}" 32
            run_experiment "${dataset}" bmlsms "${seed}" 32
        fi
    done
done
