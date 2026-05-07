#!/bin/bash
#python3 main.py --config ./config/cifar100/cifar100.py --imb-type exp --imb-factor 0.01 --loss-type mixup --mixup-beta 1.00 --train-rule None --dual-sample True --sampler-type default --dual-sampler-type balance --temp-eta 7 --temp-epsilon 0.6 --use-experts --use-experts-verbose --exp-str cifar100_100
IMB=${1}  # (0.005 0.01 0.02 0.1)
SEEDS=(28 29 32 35 47)

for SEED in ${SEEDS[@]}
do
    ARGS="
    --config ./config/cifar100/cifar100.py
    --imb-type exp
    --imb-factor ${IMB}
    --loss-type mixup
    --mixup-beta 1.00
    --train-rule None
    --dual-sample True
    --sampler-type default
    --dual-sampler-type balance
    --temp-eta 3 
    --temp-epsilon 0.6
    --use-experts
    --use-experts-verbose
    --exp-str cifar100_100
    --seed ${SEED}
    "

    python3 main.py ${ARGS}
done

