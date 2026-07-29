import math
import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class LongTailedImageDataset(Dataset):
    """List-file image dataset compatible with this repository's samplers.

    Training samples use the same six-value contract as IMBALANCECIFAR10:
    ([weak, strong], target, [dual_weak, dual_strong], dual_target,
    sample_index, dual_sample_index). Validation samples return (image, target).
    """

    def __init__(
        self,
        root,
        list_file,
        args,
        num_classes,
        train=True,
        batch_size=128,
        transform=None,
        strong_transform=None,
        target_transform=None,
        class_map=None,
        subset_thr=(100, 20),
    ):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.list_file = os.path.abspath(os.path.expanduser(list_file))
        self.args = args
        self.num_classes = num_classes
        self.cls_num = num_classes
        self.train = train
        self.batch_size = batch_size
        self.transform = transform
        self.strong_transform = strong_transform
        self.target_transform = target_transform
        self.subset_thr = list(subset_thr)
        self.single_train_view = os.environ.get(
            'BMLSBN_SINGLE_TRAIN_VIEW', '0'
        ).lower() not in ('0', 'false', 'no', 'off')

        if not os.path.isdir(self.root):
            raise FileNotFoundError("Dataset root does not exist: {}".format(self.root))
        if not os.path.isfile(self.list_file):
            raise FileNotFoundError("Dataset list file does not exist: {}".format(self.list_file))

        self.img_path, original_targets = self._read_list_file()
        self.class_map = self._build_class_map(original_targets) if class_map is None else list(class_map)
        if len(self.class_map) != self.num_classes:
            raise ValueError(
                "class_map has {} entries, expected {}".format(
                    len(self.class_map), self.num_classes
                )
            )

        self.targets = [self.class_map[target] for target in original_targets]
        counts = np.bincount(self.targets, minlength=self.num_classes)
        self.num_per_cls_dict = {
            class_index: int(count) for class_index, count in enumerate(counts)
        }
        self.img_num_per_cls = self.get_cls_num_list()
        self.class_dict = self._get_class_dict()
        self.nonempty_classes = [
            class_index
            for class_index, indices in self.class_dict.items()
            if len(indices) > 0
        ]

        self.dual_sample = bool(train and getattr(args, 'dual_sample', False))
        needs_weighted_sampling = train and (
            self.dual_sample
            or getattr(args, 'sampler_type', 'default') in ('balance', 'reverse')
            or getattr(args, 'use_soft_weight', False)
            or getattr(args, 'use_dual_soft_weight', False)
        )
        if needs_weighted_sampling:
            (
                self.class_weight,
                self.sum_weight,
                self.soft_class_weight,
                self.soft_sum_weight,
                self.num_list,
            ) = self.get_weight(
                self.get_annotations(),
                self.num_classes,
                getattr(args, 'weight_gamma', 1),
            )
            self.class_weight_bal = [
                self.sum_weight / self.num_classes for _ in range(self.num_classes)
            ]
            self.current_epoch = 0
            self.trigger = 0
            self.iter_per_epoch = math.ceil(len(self) / float(self.batch_size))
            self.instance_weights = np.ones(len(self), dtype=np.float64)

    def _read_list_file(self):
        paths = []
        targets = []
        with open(self.list_file, 'r') as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    relative_path, target_text = line.rsplit(maxsplit=1)
                    target = int(target_text)
                except (TypeError, ValueError):
                    raise ValueError(
                        "Invalid entry in {} at line {}: {!r}".format(
                            self.list_file, line_number, raw_line.rstrip('\n')
                        )
                    )
                if target < 0 or target >= self.num_classes:
                    raise ValueError(
                        "Target {} in {} at line {} is outside [0, {})".format(
                            target, self.list_file, line_number, self.num_classes
                        )
                    )
                paths.append(self._resolve_image_path(relative_path))
                targets.append(target)

        if not paths:
            raise ValueError("Dataset list file is empty: {}".format(self.list_file))
        return paths, targets

    def _resolve_image_path(self, path):
        path = os.path.expanduser(path)
        if os.path.isabs(path):
            return os.path.normpath(path)

        path = os.path.normpath(path)
        candidates = [
            os.path.join(self.root, path),
            os.path.join(self.root, 'images', path),
        ]
        root_name = os.path.basename(self.root.rstrip(os.sep))
        if path == root_name or path.startswith(root_name + os.sep):
            candidates.append(os.path.join(os.path.dirname(self.root), path))

        for candidate in candidates:
            if os.path.isfile(candidate):
                return os.path.normpath(candidate)
        return os.path.normpath(candidates[0])

    def _build_class_map(self, targets):
        counts = np.bincount(targets, minlength=self.num_classes)
        sorted_classes = np.argsort(-counts, kind='stable')
        class_map = np.empty(self.num_classes, dtype=np.int64)
        class_map[sorted_classes] = np.arange(self.num_classes)
        return class_map.tolist()

    def __len__(self):
        return len(self.targets)

    def _load_image(self, index):
        path = self.img_path[index]
        try:
            with open(path, 'rb') as handle:
                return Image.open(handle).convert('RGB')
        except FileNotFoundError:
            raise FileNotFoundError(
                "Image listed in {} was not found: {}".format(self.list_file, path)
            )

    def _choose_class(self, reverse=False, soft=False):
        weights = self.soft_class_weight if soft else self.class_weight
        if reverse:
            weights = weights[::-1]
        nonempty_weights = np.asarray(
            [weights[class_index] for class_index in self.nonempty_classes],
            dtype=np.float64,
        )
        probability = nonempty_weights / nonempty_weights.sum()
        return int(np.random.choice(self.nonempty_classes, p=probability))

    def _sample_index(self, index, sampler_type, soft=False):
        if sampler_type == 'default':
            if not soft:
                return index
            sample_class = self._choose_class(reverse=True, soft=True)
        elif sampler_type == 'balance':
            sample_class = random.choice(self.nonempty_classes)
        elif sampler_type == 'reverse':
            sample_class = self._choose_class(reverse=False, soft=soft)
        elif sampler_type == 'uniform':
            return random.randrange(len(self))
        else:
            raise ValueError("Unsupported sampler type: {}".format(sampler_type))
        return int(np.random.choice(self.class_dict[sample_class]))

    def _transform_image(self, image):
        if self.single_train_view and self.strong_transform is not None:
            strong = self.strong_transform(image)
            # All current ImageNet-LT/iNaturalist training entry points consume
            # only element 1. Reusing it avoids computing an unused weak view.
            return strong, strong
        weak = self.transform(image) if self.transform is not None else image
        strong = (
            self.strong_transform(image)
            if self.strong_transform is not None
            else weak
        )
        return weak, strong

    def __getitem__(self, index):
        if not self.train:
            image = self._load_image(index)
            target = self.targets[index]
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return image, target

        sample_index = self._sample_index(
            index,
            getattr(self.args, 'sampler_type', 'default'),
            getattr(self.args, 'use_soft_weight', False),
        )
        image = self._load_image(sample_index)
        target = self.targets[sample_index]
        image_weak, image_strong = self._transform_image(image)

        if self.dual_sample:
            dual_index = self._sample_index(
                index,
                getattr(self.args, 'dual_sampler_type', 'default'),
                getattr(self.args, 'use_dual_soft_weight', False),
            )
            dual_image = self._load_image(dual_index)
            dual_target = self.targets[dual_index]
            dual_weak, dual_strong = self._transform_image(dual_image)
        else:
            dual_index = sample_index
            dual_target = target
            dual_weak, dual_strong = image_weak, image_strong

        if self.target_transform is not None:
            target = self.target_transform(target)
            dual_target = self.target_transform(dual_target)

        return (
            [image_weak, image_strong],
            target,
            [dual_weak, dual_strong],
            dual_target,
            sample_index,
            dual_index,
        )

    def _get_class_dict(self):
        class_dict = {
            class_index: [] for class_index in range(self.num_classes)
        }
        for index, target in enumerate(self.targets):
            class_dict[target].append(index)
        return class_dict

    def get_annotations(self):
        return [{'category_id': int(target)} for target in self.targets]

    def get_cls_num_list(self):
        return [
            self.num_per_cls_dict[class_index]
            for class_index in range(self.num_classes)
        ]

    def get_weight(self, annotations, num_classes, weight_gamma):
        num_list = [0] * num_classes
        for annotation in annotations:
            num_list[annotation['category_id']] += 1
        if any(number == 0 for number in num_list):
            raise ValueError(
                "Weighted sampling requires at least one training image for every class"
            )

        max_num = max(num_list)
        weight_type = getattr(self.args, 'weight_type', 'freq')
        if weight_type == 'freq':
            class_weight = [max_num / number for number in num_list]
        elif weight_type == 'cb':
            cb_beta = getattr(self.args, 'cb_beta', 0.9999)
            if cb_beta == 0:
                cb_beta = (sum(num_list) - 1) / float(sum(num_list))
            class_weight = [
                (1 - cb_beta) / (1 - cb_beta ** number)
                for number in num_list
            ]
        else:
            raise ValueError("Unsupported weight type: {}".format(weight_type))

        gamma = float(weight_gamma)
        exponent = 0.0 if math.isinf(gamma) else 1.0 / gamma
        soft_class_weight = [weight ** exponent for weight in class_weight]
        return (
            class_weight,
            sum(class_weight),
            soft_class_weight,
            sum(soft_class_weight),
            num_list,
        )

    def update_weights(self, update_val, index, update_ema=None):
        update = update_val.detach().cpu().numpy()
        if update_ema is None:
            self.instance_weights[index] = update
        else:
            self.instance_weights[index] = (
                update_ema * self.instance_weights[index]
                + (1 - update_ema) * update
            )

    def generate_weight(self, index, args=None, sample_class=None):
        return self.instance_weights[index]
