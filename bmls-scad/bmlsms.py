"""Balanced mixed-label sampling and mixed-superclass classification.

This module ports the small, framework-independent core of BMLS+MS into SCAD:

* ``BalancedSuperclassPairSampler`` balances unordered *superclass* pairs.
* ``PairedDataset`` resolves the paired indices returned by the sampler.
* ``mixed_superclass_logits`` builds temporary mixed classifier weights from
  the original superclass head.  It does not introduce extra parameters.
"""

import math
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler


def superclass_pairs(num_superclasses, device=None):
    """Return all unordered, non-identical superclass pairs."""
    if num_superclasses < 2:
        raise ValueError("BMLS+MS requires at least two superclasses")
    return torch.tensor(
        list(combinations(range(num_superclasses), 2)),
        dtype=torch.long,
        device=device,
    )


class PairedDataset(Dataset):
    """View a regular labeled dataset through paired sample indices."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        index_a, index_b = indices
        image_a, target_a = self.dataset[index_a]
        image_b, target_b = self.dataset[index_b]
        return (image_a, image_b), (target_a, target_b)


class BalancedSuperclassPairSampler(Sampler):
    """Sample approximately equal counts of every unordered superclass pair.

    Fine-class targets are mapped to their fixed SCAD superclass labels before
    pair construction. Samples inside each superclass are drawn cyclically
    from reshuffled index arrays, matching the balancing behavior of BMLS.
    """

    def __init__(
        self,
        fine_targets,
        coarse_labels,
        num_samples=None,
        seed=0,
        num_replicas=1,
        rank=0,
    ):
        fine_targets = np.asarray(fine_targets, dtype=np.int64)
        coarse_labels = np.asarray(coarse_labels, dtype=np.int64)
        if fine_targets.ndim != 1:
            raise ValueError("fine_targets must be a one-dimensional sequence")
        if fine_targets.size == 0:
            raise ValueError("fine_targets must not be empty")
        if fine_targets.min() < 0 or fine_targets.max() >= len(coarse_labels):
            raise ValueError("fine_targets contain labels outside coarse_labels")
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("invalid distributed sampler rank configuration")

        self.super_targets = coarse_labels[fine_targets]
        self.num_superclasses = int(coarse_labels.max()) + 1
        self.pairs = list(combinations(range(self.num_superclasses), 2))
        if not self.pairs:
            raise ValueError("BMLS+MS requires at least two superclasses")

        self.indices_by_superclass = [
            np.flatnonzero(self.super_targets == superclass)
            for superclass in range(self.num_superclasses)
        ]
        empty = [
            superclass
            for superclass, indices in enumerate(self.indices_by_superclass)
            if len(indices) == 0
        ]
        if empty:
            raise ValueError(
                "labeled data contain no examples for superclasses "
                + ", ".join(map(str, empty))
            )

        requested_samples = len(fine_targets) if num_samples is None else int(num_samples)
        if requested_samples < 1:
            raise ValueError("num_samples must be positive")
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.num_samples = int(math.ceil(requested_samples / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    @staticmethod
    def _draw(indices, count, rng):
        """Draw ``count`` indices, reshuffling whenever the source wraps."""
        output = np.empty(count, dtype=np.int64)
        offset = 0
        while offset < count:
            shuffled = rng.permutation(indices)
            take = min(len(shuffled), count - offset)
            output[offset:offset + take] = shuffled[:take]
            offset += take
        return output

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        pair_count = len(self.pairs)
        counts = np.full(pair_count, self.total_size // pair_count, dtype=np.int64)
        remainder = self.total_size % pair_count
        if remainder:
            counts[rng.permutation(pair_count)[:remainder]] += 1

        sampled_pairs = []
        for (super_a, super_b), count in zip(self.pairs, counts):
            if count == 0:
                continue
            indices_a = self._draw(self.indices_by_superclass[super_a], int(count), rng)
            indices_b = self._draw(self.indices_by_superclass[super_b], int(count), rng)
            sampled_pairs.extend(zip(indices_a.tolist(), indices_b.tolist()))

        rng.shuffle(sampled_pairs)
        sampled_pairs = sampled_pairs[self.rank:self.total_size:self.num_replicas]
        if len(sampled_pairs) != self.num_samples:
            raise RuntimeError("distributed BMLS sampler produced an invalid shard")
        return iter(sampled_pairs)


def mixed_superclass_logits(features, classifier, targets_a, targets_b, lam):
    """Classify mixed features with dynamically composed superclass weights.

    Returns logits over all unordered superclass pairs and the singleton target
    index for each input pair.
    """
    if classifier.weight.ndim != 2:
        raise ValueError("classifier must expose a two-dimensional weight")

    num_superclasses = classifier.weight.shape[0]
    pairs = superclass_pairs(num_superclasses, device=features.device)
    targets_a = targets_a.to(device=features.device, dtype=torch.long)
    targets_b = targets_b.to(device=features.device, dtype=torch.long)
    low = torch.minimum(targets_a, targets_b)
    high = torch.maximum(targets_a, targets_b)
    if torch.any(low == high):
        raise ValueError("BMLS superclass pairs must contain different labels")

    lam = torch.as_tensor(lam, dtype=features.dtype, device=features.device)
    weight = classifier.weight
    mixed_weight = lam * weight[pairs[:, 0]] + (1.0 - lam) * weight[pairs[:, 1]]

    mixed_bias = None
    if classifier.bias is not None:
        mixed_bias = (
            lam * classifier.bias[pairs[:, 0]]
            + (1.0 - lam) * classifier.bias[pairs[:, 1]]
        )

    pair_lookup = torch.full(
        (num_superclasses, num_superclasses),
        -1,
        dtype=torch.long,
        device=features.device,
    )
    pair_lookup[pairs[:, 0], pairs[:, 1]] = torch.arange(
        len(pairs), device=features.device
    )
    pair_targets = pair_lookup[low, high]
    if torch.any(pair_targets < 0):
        raise ValueError("failed to map one or more superclass pairs")

    return F.linear(features, mixed_weight, mixed_bias), pair_targets
