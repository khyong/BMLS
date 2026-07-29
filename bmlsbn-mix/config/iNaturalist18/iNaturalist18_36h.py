"""Four-epoch iNaturalist comparison matching the active 15-epoch run.

The LR milestones deliberately remain at 9 and 12.  Consequently, the first
four epochs have exactly the same optimizer schedule as the interrupted
15-epoch experiment and its checkpoints can be compared at equal compute.
"""

from config.iNaturalist18._base import build_config


config = build_config(use_norm='fc')
config['training_opt']['num_epochs'] = 4
config['optimizer']['scheduler_params']['warmup_epoch'] = 1
config['optimizer']['scheduler_params']['step'] = [9, 12]
