import os

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from data.autoaugment import autoCIFARPolicy, autoImageNetPolicy, autoSVHNPolicy
from data.imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from data.imbalance_imagenet import ImageNetLT
from data.ina2018 import iNa2018
from data.randaugment import randCIFARPolicy, randImageNetPolicy, randSVHNPolicy
from data.sampler import (
    BalancedMixedLabelSampler,
    DoubleBalancedMixedLabelSampler,
    DoubleMixedLabelDataset,
    DualBalancedMixedLabelSampler,
    DualMixedLabelDataset,
    MixedLabelDataset,
)


dataset_statistics = {
    'cifar10': {
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2471, 0.2435, 0.2616),
    },
    'cifar100': {
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761),
    },
    'imagenetLT': {
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
    },
    'iNaturalist18': {
        'mean': (0.466, 0.471, 0.380),
        'std': (0.195, 0.194, 0.192),
    },
    'placesLT': {
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
    },
}

num_classes_list = {
    'cifar10': 10,
    'cifar100': 100,
    'placesLT': 365,
    'imagenetLT': 1000,
    'iNaturalist18': 8142,
}


def get_num_classes(dataset):
    return num_classes_list[dataset]


def load_data(
    data_root,
    dataset,
    batch_size,
    test_open=False,
    num_workers=4,
    ngpus_per_node=None,
    args=None,
):
    del test_open
    if args is None:
        raise ValueError('args is required to construct the training dataset')
    if dataset not in dataset_statistics:
        raise ValueError("Unsupported dataset: {}".format(dataset))

    # Large images and the mixed-label datasets make each training sample read
    # up to three files.  Allow node-specific tuning without editing configs,
    # and avoid DataLoader's default two-batch prefetch burst on slower disks.
    num_workers = int(os.environ.get('BMLSBN_NUM_WORKERS', num_workers))
    prefetch_factor = int(os.environ.get('BMLSBN_PREFETCH_FACTOR', 1))
    if num_workers < 0:
        raise ValueError('BMLSBN_NUM_WORKERS must be non-negative')
    if prefetch_factor < 1:
        raise ValueError('BMLSBN_PREFETCH_FACTOR must be at least 1')

    statistics = dataset_statistics[dataset]
    dataset_mean = statistics['mean']
    dataset_std = statistics['std']
    num_classes = num_classes_list[dataset]

    if dataset in ('cifar10', 'cifar100'):
        root = os.path.join(os.path.expanduser(data_root), dataset)
        train_transform_s = get_cifar_transform(
            'strong_train', args.augmentation, dataset_mean, dataset_std
        )
        train_transform_w = get_cifar_transform(
            'weak_train', args.augmentation, dataset_mean, dataset_std
        )
        val_transform = get_cifar_transform(
            'val', args.augmentation, dataset_mean, dataset_std
        )
    elif dataset == 'imagenetLT':
        root = os.path.abspath(os.path.expanduser(data_root))
        train_transform_s = get_imagenet_transforms(
            'strong_train', args.augmentation, dataset_mean, dataset_std
        )
        train_transform_w = get_imagenet_transforms(
            'weak_train', args.augmentation, dataset_mean, dataset_std
        )
        val_transform = get_imagenet_transforms(
            'val', args.augmentation, dataset_mean, dataset_std
        )
    elif dataset == 'iNaturalist18':
        root = os.path.abspath(os.path.expanduser(data_root))
        train_transform_s = get_iNaturalist_transforms(
            'strong_train', args.augmentation, dataset_mean, dataset_std
        )
        train_transform_w = get_iNaturalist_transforms(
            'weak_train', args.augmentation, dataset_mean, dataset_std
        )
        val_transform = get_iNaturalist_transforms(
            'val', args.augmentation, dataset_mean, dataset_std
        )
    else:
        raise ValueError("Dataset loading is not implemented for {}".format(dataset))

    if dataset == 'cifar10':
        train_dataset = IMBALANCECIFAR10(
            root=root,
            args=args,
            imb_type=args.imb_type,
            imb_factor=args.imb_factor,
            rand_number=args.seed,
            train=True,
            batch_size=batch_size,
            download=True,
            transform=train_transform_w,
            strong_transform=train_transform_s,
        )
        val_dataset = datasets.CIFAR10(
            root=root, train=False, download=True, transform=val_transform
        )
    elif dataset == 'cifar100':
        train_dataset = IMBALANCECIFAR100(
            root=root,
            args=args,
            imb_type=args.imb_type,
            imb_factor=args.imb_factor,
            rand_number=args.seed,
            train=True,
            batch_size=batch_size,
            download=True,
            transform=train_transform_w,
            strong_transform=train_transform_s,
        )
        val_dataset = datasets.CIFAR100(
            root=root, train=False, download=True, transform=val_transform
        )
    elif dataset == 'imagenetLT':
        train_dataset = ImageNetLT(
            root=root,
            args=args,
            train=True,
            batch_size=batch_size,
            transform=train_transform_w,
            strong_transform=train_transform_s,
        )
        val_dataset = ImageNetLT(
            root=root,
            args=args,
            train=False,
            batch_size=batch_size,
            transform=val_transform,
            class_map=train_dataset.class_map,
        )
    else:
        train_dataset = iNa2018(
            root=root,
            args=args,
            train=True,
            batch_size=batch_size,
            transform=train_transform_w,
            strong_transform=train_transform_s,
        )
        val_dataset = iNa2018(
            root=root,
            args=args,
            train=False,
            batch_size=batch_size,
            transform=val_transform,
            class_map=train_dataset.class_map,
        )

    train_sampler = None
    if args.train_rule == 'BMLS':
        train_sampler = BalancedMixedLabelSampler(
            train_dataset,
            num_classes=num_classes,
            num_class_list=train_dataset.get_cls_num_list(),
        )
        train_dataset = MixedLabelDataset(train_dataset)
    elif args.train_rule == 'DualBMLS':
        train_sampler = DualBalancedMixedLabelSampler(
            train_dataset,
            num_classes=num_classes,
            num_class_list=train_dataset.get_cls_num_list(),
        )
        train_dataset = DualMixedLabelDataset(train_dataset)
    elif args.train_rule == 'DoubleBMLS':
        train_sampler = DoubleBalancedMixedLabelSampler(
            train_dataset,
            num_classes=num_classes,
            num_class_list=train_dataset.get_cls_num_list(),
        )
        train_dataset = DoubleMixedLabelDataset(train_dataset)

    # Long-tailed circular pair sampling must consume its final partial batch;
    # otherwise those pairs are skipped permanently when the next epoch moves
    # to the next segment of the C^2 cycle. The iNaturalist remainder is safe
    # for DataParallel (9 samples at batch 256, 265 at batch 1024).
    drop_last = bool(
        ngpus_per_node
        and ngpus_per_node > 1
        and dataset in ('cifar10', 'cifar100')
    )
    loader_worker_options = {
        'num_workers': num_workers,
        'persistent_workers': num_workers > 0,
    }
    if num_workers > 0:
        loader_worker_options['prefetch_factor'] = prefetch_factor

    print(
        'DataLoader: workers={}, prefetch_factor={}'.format(
            num_workers, prefetch_factor if num_workers > 0 else 'disabled'
        )
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=drop_last,
        **loader_worker_options,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=100,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        **loader_worker_options,
    )
    print(
        'Training loader: samples={}, global_batch_size={}, batches={}'.format(
            len(train_dataset), batch_size, len(train_loader)
        )
    )
    return train_loader, val_loader


class noPolicy(object):
    def __call__(self, img):
        return img


def get_augment(dataset, aug_method):
    policies = {
        'svhn': {
            'autoaugment': autoSVHNPolicy,
            'randaugment': randSVHNPolicy,
            'default': noPolicy,
        },
        'cifar': {
            'autoaugment': autoCIFARPolicy,
            'randaugment': randCIFARPolicy,
            'default': noPolicy,
        },
        'imagenetLT': {
            'autoaugment': autoImageNetPolicy,
            'randaugment': randImageNetPolicy,
            'default': noPolicy,
        },
        'iNaturalist18': {
            'autoaugment': autoImageNetPolicy,
            'randaugment': randImageNetPolicy,
            'default': noPolicy,
        },
        'placesLT': {
            'autoaugment': autoImageNetPolicy,
            'randaugment': randImageNetPolicy,
            'default': noPolicy,
        },
    }
    try:
        return policies[dataset][aug_method]
    except KeyError:
        raise ValueError(
            "Unsupported augmentation {!r} for dataset {!r}".format(
                aug_method, dataset
            )
        )


def get_cifar_transform(split, aug_method, dataset_mean, dataset_std):
    data_transforms = {
        'strong_train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            get_augment('cifar', aug_method)(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std),
        ]),
        'weak_train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std),
        ]),
    }
    return data_transforms[split]


def _get_large_image_transforms(
    dataset, split, aug_method, dataset_mean, dataset_std
):
    train_prefix = [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std),
    ]
    data_transforms = {
        'strong_train': transforms.Compose(
            train_prefix + [get_augment(dataset, aug_method)()] + normalize
        ),
        'weak_train': transforms.Compose(train_prefix + normalize),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ] + normalize),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ] + normalize),
    }
    return data_transforms[split]


def get_imagenet_transforms(split, aug_method, dataset_mean, dataset_std):
    return _get_large_image_transforms(
        'imagenetLT', split, aug_method, dataset_mean, dataset_std
    )


def get_iNaturalist_transforms(split, aug_method, dataset_mean, dataset_std):
    return _get_large_image_transforms(
        'iNaturalist18', split, aug_method, dataset_mean, dataset_std
    )


def get_places_transforms(split, aug_method, dataset_mean, dataset_std):
    return _get_large_image_transforms(
        'placesLT', split, aug_method, dataset_mean, dataset_std
    )
