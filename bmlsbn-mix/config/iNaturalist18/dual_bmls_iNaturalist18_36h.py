"""Four-epoch dual-BMLS linear-classifier configuration."""

from config.iNaturalist18._base import build_config


config = build_config(use_norm='fc')
config['training_opt']['num_epochs'] = 4
config['optimizer']['scheduler_params']['warmup_epoch'] = 1
config['optimizer']['scheduler_params']['step'] = [9, 12]
