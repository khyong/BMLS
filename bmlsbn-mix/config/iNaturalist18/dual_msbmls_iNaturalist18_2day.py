"""Matched, time-budgeted Dual-MS-BMLS iNaturalist 2018 configuration."""

from config.iNaturalist18._base import build_config


config = build_config(use_norm='dual_ms')
config['training_opt']['num_epochs'] = 15
config['optimizer']['scheduler_params']['warmup_epoch'] = 1
config['optimizer']['scheduler_params']['step'] = [9, 12]
