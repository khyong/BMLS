#!/bin/bash
SEED=${1}

ARGS="
--config ./config/cifar10/dual_msbmls_cifar10.py
--imb-type exp
--imb-factor 0.01
--loss-type dual_mixup
--mixup-beta 1.00
--train-rule DualBMLS
--dual-sample False
--sampler-type default
--temp-eta 3 
--temp-epsilon 0.6
--use-experts
--use-experts-verbose
--exp-str dual_msbmls_cifar10_100
--seed ${SEED}
"

python3 dual_msbmls_main.py ${ARGS}

