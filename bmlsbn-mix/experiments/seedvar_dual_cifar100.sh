#!/bin/bash
IMB=${1}  # (0.005 0.01 0.02 0.1)
SEEDS=(28 29 32 35 47)

for SEED in ${SEEDS[@]}
do
    ARGS="
    --config ./config/cifar100/dual_msbmls_cifar100.py
    --imb-type exp
    --imb-factor ${IMB}
    --loss-type dual_mixup
    --mixup-beta 1.00
    --train-rule DualBMLS
    --dual-sample False
    --sampler-type default
    --temp-eta 3 
    --temp-epsilon 0.6
    --use-experts
    --use-experts-verbose
    --exp-str dual_msbmls_cifar100_100
    --seed ${SEED}
    "

    python3 dual_msbmls_main.py ${ARGS}
done

