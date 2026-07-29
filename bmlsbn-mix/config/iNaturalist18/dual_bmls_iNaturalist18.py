from config.iNaturalist18._base import build_config


# Dual random-CB/circular-BMLS model with ordinary linear classifiers.
config = build_config(use_norm='fc')
