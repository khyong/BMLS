"""Four-epoch Dual-MS-BMLS config matching the active 15-epoch run."""

from config.iNaturalist18._base import build_config


config = build_config(use_norm='dual_ms')
config['training_opt']['num_epochs'] = 4
config['optimizer']['scheduler_params']['warmup_epoch'] = 1
config['optimizer']['scheduler_params']['step'] = [9, 12]
