#!/bin/bash
#python3 main.py --config ./config/cifar10/cifar10.py --imb-type exp --imb-factor 0.01 --loss-type mixup --mixup-beta 1.00 --train-rule None --dual-sample True --sampler-type default --dual-sampler-type balance --temp-eta 3 --temp-epsilon 0.6 --use-experts --use-experts-verbose --exp-str cifar10_100

ARGS="
--config ./config/cifar10/bmls_cifar10.py
--imb-type exp
--imb-factor 0.01
--loss-type mixup
--mixup-beta 1.00
--train-rule BMLS
--dual-sample False
--sampler-type default
--temp-eta 3 
--temp-epsilon 0.6
--use-experts
--use-experts-verbose
--exp-str bmls_cifar10_100
"

python3 bmls_main.py ${ARGS}

