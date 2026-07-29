def build_config(use_norm, extended=False, log_root='output'):
    """Build an ImageNet-LT configuration.

    The standard schedule runs for 90 epochs and decays the learning rate at
    epochs 60 and 80. The extended schedule runs for 200 epochs and decays it
    at epochs 120 and 160.
    """
    training_opt = {
        'dataset': 'imagenetLT',
        'num_classes': 1000,
        # Global batch size 256; DataParallel splits it across visible GPUs.
        'batch_size': 256,
        # Each dual-mix sample opens three images. A small worker count avoids
        # random-read contention on /data; override with BMLSBN_NUM_WORKERS.
        'num_workers': 2,
        'num_epochs': 200 if extended else 90,
        'start_epoch': 0,
        'print_freq': 10,
        'accumulation_steps': 1,
        'log_root': log_root,
        'data_root': '/data/hoyong/imagenet',
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
            'lr': 0.2,
            'momentum': 0.9,
            'weight_decay': 2e-4,
            'nesterov': True,
        },
        'scheduler': 'step',
        'scheduler_params': {
            'step': [120, 160] if extended else [60, 80],
            'gamma': 0.1,
            'warmup_epoch': 5,
        },
    }

    algorithm = {
        'mixup_alpha': 0.2,
        'temp_eta': 9,
        'temp_epsilon': 0.2,
        'rebalance_gamma': float('inf'),
    }

    return {
        'training_opt': training_opt,
        'networks': networks,
        'optimizer': optimizer,
        'algorithm': algorithm,
    }
