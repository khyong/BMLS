import os


def build_config(use_norm, log_root='output'):
    """Build an iNaturalist 2018 configuration."""
    training_opt = {
        'dataset': 'iNaturalist18',
        'num_classes': 8142,
        # Global batch size 256; DataParallel splits it across visible GPUs.
        'batch_size': 768,
        # Each dual-mix sample opens three images. A small worker count avoids
        # random-read contention on /data; override with BMLSBN_NUM_WORKERS.
        'num_workers': 2,
        'num_epochs': 200,
        'start_epoch': 0,
        'print_freq': 10,
        'accumulation_steps': 1,
        'log_root': log_root,
        # Keep /data as the fallback, but permit staging the JPEGs on SSD/NVMe.
        'data_root': os.environ.get(
            'BMLSBN_INAT_ROOT', '/data/hoyong/iNaturalist2018'
        ),
        'imb_type': 'exp',
        # None enables DataParallel across every GPU in CUDA_VISIBLE_DEVICES.
        'gpu': None,
    }

    networks = {
        'arch': 'resnet50',
        'param': {
            'num_classes': training_opt['num_classes'],
            'use_norm': use_norm,
            'use_block': True,
        },
    }

    optimizer = {
        'type': 'SGD',
        'optim_params': {
            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'nesterov': True,
        },
        'scheduler': 'step',
        'scheduler_params': {
            'step': [120, 160],
            'gamma': 0.1,
            'warmup_epoch': 5,
        },
    }

    algorithm = {
        'mixup_alpha': 0.2,
        'temp_eta': 10,
        'temp_epsilon': 2e-2,
        'rebalance_gamma': float('inf'),
    }

    return {
        'training_opt': training_opt,
        'networks': networks,
        'optimizer': optimizer,
        'algorithm': algorithm,
    }
