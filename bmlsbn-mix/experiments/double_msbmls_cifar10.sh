#!/bin/bash

ARGS="
--config ./config/cifar10/double_msbmls_cifar10.py
--imb-type exp
--imb-factor 0.01
--loss-type double_mixup
--mixup-beta 1.00
--train-rule DoubleBMLS
--dual-sample False
--sampler-type default
--temp-eta 3 
--temp-epsilon 0.6
--use-experts
--use-experts-verbose
--exp-str double_msbmls_cifar10_100
"

python3 double_msbmls_main.py ${ARGS}

