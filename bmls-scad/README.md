## Requirements

- matplotlib
- tqdm
- torch
- tensorboard

## Dataset

The directory structure for datasets looks like:
```
datasets
├── cifar-10
├── cifar-100
├── stl-10
├── imagenet32
└── imagenet64
```


## Usage
if you want to run the code only , fixmatch based method then run fixmatchSCAD.py
if you want to run the code, SCAD based method with ACR then run acrwithSCAD.py

For example:

For consistent:
```
python acrwithSCAD.py --dataset cifar10 --num-max 500 --num-max-u 4000 --arch wideresnet --batch-size 64 --lr 0.03 --seed 0 --imb-ratio-label 100 --imb-ratio-unlabel 100 --ema-u 0.99 --out out/cifar-10/N500_M4000/consistent
```

For uniform:
```
python acrwithSCAD.py --dataset cifar10 --num-max 500 --num-max-u 4000 --arch wideresnet --batch-size 64 --lr 0.03 --seed 0 --imb-ratio-label 100 --imb-ratio-unlabel 1 --ema-u 0.99 --out out/cifar-10/N500_M4000/uniform
```

For reversed:
```
python acrwithSCAD.py --dataset cifar10 --num-max 500 --num-max-u 4000 --arch wideresnet --batch-size 64 --lr 0.03 --seed 0 --imb-ratio-label 100 --imb-ratio-unlabel 100 --ema-u 0.99 --flag-reverse-LT 1 --out out/cifar-10/N500_M4000/reversed
```

## BMLS+MS superclass balancing

`acrwithSCAD.py` now supports two balancing methods for the second superclass
head (`sfc1`):

- `--superclass-method scad` keeps the original superclass logit-adjustment
  loss. This is the default.
- `--superclass-method bmlsms` replaces the labeled `sfc1` logit-adjustment
  term with superclass-aware Balanced Mixed-Label Sampling and a mixed
  classifier. The original `sfc` path and the unlabeled `sfc1` consistency
  loss remain unchanged.

Example:

```bash
python acrwithSCAD.py \
  --dataset cifar10 \
  --superclass-method bmlsms \
  --bmlsms-alpha 1.0 \
  --bmlsms-lambda 1.0 \
  --out out/cifar10/bmlsms
```

The mixed classifier contains no additional learned parameters. For a
superclass pair `(a, b)`, it derives its temporary weight as
`lambda * W[a] + (1 - lambda) * W[b]` from `sfc1`.

Run the CPU pilot tests with:

```bash
python -m unittest -v tests.test_bmlsms
```

The pilots exercise the checked-in superclass maps and the repository
WideResNet for CIFAR-10, CIFAR-100, Small ImageNet 32×32, and Small ImageNet
64×64. They verify pair balancing, mixed-logit dimensions, finite loss,
gradient flow through the backbone and `sfc1`, and an optimizer update.

To compare original SCAD and BMLSMS-SCAD on all three datasets:

```bash
DATA_ROOT=/path/to/datasets \
SEEDS="0 1 2" \
./scripts/compare_scad_bmlsms.sh
```

The comparison defaults to the consistent long-tailed setting and writes
results under `out/comparison`. Both Small ImageNet modes (`imagenet32` and
`imagenet64`) run by default and are stored under `smallimagenet/img32` and
`smallimagenet/img64`. Environment variables such as `TOTAL_STEPS`,
`EVAL_STEP`, `BATCH_SIZE`, `CIFAR10_NUM_MAX`, `CIFAR100_NUM_MAX`,
`LABELED_IMB_RATIO`, and `UNLABELED_IMB_RATIO` can override the defaults.
Set `SMALLIMAGENET_IMG_SIZES="32"` or `"64"` to run only one mode.

### FixMatchSCAD variant

The same opt-in BMLS+MS superclass objective is available in
`fixmatchSCAD.py`:

```bash
python fixmatchSCAD.py \
  --dataset cifar10 \
  --superclass-method bmlsms \
  --bmlsms-alpha 1.0 \
  --bmlsms-lambda 1.0 \
  --out out/fixmatch-cifar10/bmlsms
```

FixMatchSCAD keeps its original `sfc` supervised and unlabeled consistency
losses. BMLS+MS trains the otherwise-unused `sfc1` head and shares its
gradients with the backbone. In BMLS+MS mode, post-hoc superclass adjustment
uses the learned `sfc1` inference weights. The original behavior remains the
default.

Run the FixMatch comparison with:

```bash
DATA_ROOT=/path/to/datasets \
SEEDS="0 1 2" \
./scripts/compare_fixmatch_scad_bmlsms.sh
```

This launcher also runs both Small ImageNet resolutions by default.

### Dataset-specific comparison scripts

Each dataset can be launched separately.

ACR-SCAD versus BMLSMS-SCAD:

```bash
./scripts/compare_scad_cifar10.sh
./scripts/compare_scad_cifar100.sh
./scripts/compare_scad_smallimagenet.sh
```

FixMatchSCAD versus BMLSMS-FixMatchSCAD:

```bash
./scripts/compare_fixmatch_cifar10.sh
./scripts/compare_fixmatch_cifar100.sh
./scripts/compare_fixmatch_smallimagenet.sh
```

All environment overrides accepted by the combined launchers remain
available. For example:

```bash
DATA_ROOT=/path/to/datasets \
SEEDS="0 1 2" \
TOTAL_STEPS=250000 \
./scripts/compare_scad_cifar10.sh
```

The Small ImageNet scripts run both 32×32 and 64×64 modes. Select one with
`SMALLIMAGENET_IMG_SIZES="32"` or `SMALLIMAGENET_IMG_SIZES="64"`.

## acknowlegement
this code is based on ACR and fixmatch-pytorch.
