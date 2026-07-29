#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3
export BMLSBN_AMP=1

export BMLSBN_INAT_ROOT=/home/hoyong/datasets/ina2018_256x256
export BMLSBN_INAT_LIST_ROOT=/home/hoyong/datasets/data_txt
export BMLSBN_NUM_WORKERS=2
export BMLSBN_PREFETCH_FACTOR=1

python -c '
import torch
print("Visible GPUs:", torch.cuda.device_count())
assert torch.cuda.device_count() == 4
'

bash experiments/dual_msbmls_ina2018.sh
#bash experiments/ina2018.sh
