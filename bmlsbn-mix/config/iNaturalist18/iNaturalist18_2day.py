"""Matched, time-budgeted iNaturalist 2018 configuration.

This keeps the full dataset and batch size while scaling the 200-epoch
schedule to 15 epochs for a two-job comparison on four RTX 3090 GPUs.
"""

from config.iNaturalist18._base import build_config


config = build_config(use_norm='fc')
config['training_opt']['num_epochs'] = 15
config['optimizer']['scheduler_params']['warmup_epoch'] = 1
config['optimizer']['scheduler_params']['step'] = [9, 12]
