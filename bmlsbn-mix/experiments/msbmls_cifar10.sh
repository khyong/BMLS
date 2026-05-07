#!/bin/bash

ARGS="
--config ./config/cifar10/msbmls_cifar10.py
--imb-type exp
--imb-factor 0.01
--loss-type CE
--mixup-beta 1.00
--train-rule BMLS
--dual-sample False
--sampler-type default
--temp-eta 3 
--temp-epsilon 0.6
--use-experts
--use-experts-verbose
--exp-str msbmls_cifar10_100
"

python3 msbmls_main.py ${ARGS}

