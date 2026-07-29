#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PILOT_DATASET="iNaturalist18"
export PILOT_DATA_ROOT="${1:-/data/hoyong/iNaturalist2018}"
export PILOT_CONFIG="${PILOT_CONFIG:-./config/iNaturalist18/iNaturalist18.py}"
export PILOT_TRAIN_RULE="${PILOT_TRAIN_RULE:-None}"
export PILOT_DUAL_SAMPLE="${PILOT_DUAL_SAMPLE:-true}"
export PILOT_DUAL_SAMPLER="${PILOT_DUAL_SAMPLER:-reverse}"
export PILOT_BATCH_SIZE="${PILOT_BATCH_SIZE:-2}"

python3 - <<'PY'
import os
from types import SimpleNamespace

import torch

import models
from data import load_data
from utils import source_import


dataset = os.environ['PILOT_DATASET']
data_root = os.environ['PILOT_DATA_ROOT']
config_path = os.environ['PILOT_CONFIG']
train_rule = os.environ['PILOT_TRAIN_RULE']
batch_size = int(os.environ['PILOT_BATCH_SIZE'])
dual_sample = os.environ['PILOT_DUAL_SAMPLE'].lower() == 'true'

config = source_import(config_path).config
expected_classes = config['training_opt']['num_classes']
args = SimpleNamespace(
    augmentation='default',
    imb_type='exp',
    imb_factor=0.01,
    seed=1,
    train_rule=train_rule,
    sampler_type='default',
    dual_sample=dual_sample,
    dual_sampler_type=os.environ['PILOT_DUAL_SAMPLER'],
    weight_gamma=float('inf'),
    use_soft_weight=False,
    use_dual_soft_weight=dual_sample,
    weight_type='freq',
    cb_beta=0.9999,
)

print('Pilot dataset: {}'.format(dataset))
print('Data root: {}'.format(data_root))
print('Config: {}'.format(config_path))
print('Train rule: {}'.format(train_rule))

train_loader, val_loader = load_data(
    data_root=data_root,
    dataset=dataset,
    batch_size=batch_size,
    num_workers=0,
    ngpus_per_node=1,
    args=args,
)

train_batch = next(iter(train_loader))
val_images, val_targets = next(iter(val_loader))
base_dataset = getattr(train_loader.dataset, 'dataset', train_loader.dataset)
class_counts = base_dataset.get_cls_num_list()

assert len(class_counts) == expected_classes
assert sum(class_counts) == len(base_dataset)
assert min(class_counts) > 0
assert val_images.ndim == 4 and tuple(val_images.shape[1:]) == (3, 224, 224)
assert val_targets.ndim == 1

model = models.__dict__[config['networks']['arch']](
    **config['networks']['param']
)
device_name = os.environ.get(
    'PILOT_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'
)
device = torch.device(device_name)
model = model.to(device)
model.eval()

with torch.no_grad():
    if train_rule == 'DualBMLS':
        _, _, label_pairs = train_loader.sampler.get_palette()
        model.linear_rb.init_lbl_mix2new(label_pairs)
        input_cb = train_batch[0].to(device)
        input_a = train_batch[1].to(device)
        input_b = train_batch[2].to(device)
        target_a = train_batch[4].to(device)
        target_b = train_batch[5].to(device)
        lam = 0.3
        input_rb = lam * input_a + (1 - lam) * input_b
        _, mixed_targets = model.linear_rb.init_mixed_weights_batch(
            lam, target_a, target_b
        )
        output_cb = model(input_cb, classifier_cb=True)
        output_rb = model(input_rb, classifier_rb=True, train=True)
    else:
        input_cb = train_batch[0][1].to(device)
        input_rb = train_batch[2][1].to(device)
        output_cb = model(input_cb, classifier_cb=True)
        output_rb = model(input_rb, classifier_rb=True)

assert output_cb.shape == (batch_size, expected_classes)
if train_rule == 'DualBMLS':
    assert output_rb.shape == (
        batch_size, int(mixed_targets.max().item()) + 1
    )
else:
    assert output_rb.shape == (batch_size, expected_classes)
assert torch.isfinite(output_cb).all()
assert torch.isfinite(output_rb).all()

print('Train images: {}'.format(len(base_dataset)))
print('Validation images: {}'.format(len(val_loader.dataset)))
print('Model outputs: {} / {}'.format(
    tuple(output_cb.shape), tuple(output_rb.shape)
))
print('PILOT PASSED')
PY
