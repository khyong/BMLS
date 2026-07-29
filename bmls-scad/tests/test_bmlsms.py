import json
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from bmlsms import (
    BalancedSuperclassPairSampler,
    PairedDataset,
    mixed_superclass_logits,
    superclass_pairs,
)
from models.wideresnet import WideResNet


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "cifar10": (10, 3),
    "cifar100": (100, 25),
    "smallimagenet": (127, 32),
}


class SyntheticDataset(Dataset):
    def __init__(self, num_classes):
        self.targets = np.repeat(np.arange(num_classes), 2)
        generator = torch.Generator().manual_seed(num_classes)
        self.images = torch.randn(
            len(self.targets), 3, 8, 8, generator=generator
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.images[index], int(self.targets[index])


class TinyModel(nn.Module):
    def __init__(self, num_superclasses):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 16),
            nn.ReLU(),
        )
        self.sfc1 = nn.Linear(16, num_superclasses)

    def forward(self, images):
        return self.backbone(images)


class BMLSMSPilotTest(unittest.TestCase):
    def _load_coarse_labels(self, dataset_name):
        path = ROOT / "labels" / f"{dataset_name}_superclass_labels.json"
        with path.open() as handle:
            groups = json.load(handle)
        coarse = np.empty(sum(len(group) for group in groups), dtype=np.int64)
        for superclass, group in enumerate(groups):
            coarse[group] = superclass
        return groups, coarse

    def _run_dataset_gradient_pilot(self, dataset_name):
        """Run one paired optimization step for one dataset configuration."""
        num_classes, num_superclasses = DATASETS[dataset_name]
        groups, coarse = self._load_coarse_labels(dataset_name)
        self.assertEqual(len(coarse), num_classes)
        self.assertEqual(len(groups), num_superclasses)
        self.assertEqual(
            len(superclass_pairs(num_superclasses)),
            num_superclasses * (num_superclasses - 1) // 2,
        )

        dataset = SyntheticDataset(num_classes)
        sampler = BalancedSuperclassPairSampler(
            dataset.targets,
            coarse,
            num_samples=max(16, num_superclasses),
            seed=7,
        )
        loader = DataLoader(
            PairedDataset(dataset),
            batch_size=8,
            sampler=sampler,
        )
        (images_a, images_b), (targets_a, targets_b) = next(iter(loader))
        super_a = torch.from_numpy(coarse[targets_a.numpy()])
        super_b = torch.from_numpy(coarse[targets_b.numpy()])
        self.assertTrue(torch.all(super_a != super_b))

        model = TinyModel(num_superclasses)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        lam = 0.4
        features = model(lam * images_a + (1.0 - lam) * images_b)
        logits, pair_targets = mixed_superclass_logits(
            features,
            model.sfc1,
            super_a,
            super_b,
            lam,
        )
        self.assertEqual(
            logits.shape,
            (len(images_a), num_superclasses * (num_superclasses - 1) // 2),
        )
        loss = F.cross_entropy(logits, pair_targets)
        optimizer.zero_grad()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.backbone[1].weight.grad)
        self.assertIsNotNone(model.sfc1.weight.grad)
        self.assertTrue(torch.isfinite(model.sfc1.weight.grad).all())
        before = model.sfc1.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.sfc1.weight))

    def test_cifar10_pilot(self):
        self._run_dataset_gradient_pilot("cifar10")

    def test_cifar100_pilot(self):
        self._run_dataset_gradient_pilot("cifar100")

    def test_smallimagenet_pilot(self):
        self._run_dataset_gradient_pilot("smallimagenet")

    def test_sampler_balances_pairs_when_epoch_covers_pair_space(self):
        _, coarse = self._load_coarse_labels("cifar10")
        dataset = SyntheticDataset(10)
        sampler = BalancedSuperclassPairSampler(
            dataset.targets,
            coarse,
            num_samples=30,
            seed=3,
        )
        counts = {}
        for index_a, index_b in sampler:
            pair = tuple(sorted((coarse[dataset.targets[index_a]], coarse[dataset.targets[index_b]])))
            counts[pair] = counts.get(pair, 0) + 1
        self.assertEqual(set(counts.values()), {10})

    def test_distributed_shards_are_deterministic(self):
        _, coarse = self._load_coarse_labels("cifar10")
        dataset = SyntheticDataset(10)
        samplers = [
            BalancedSuperclassPairSampler(
                dataset.targets,
                coarse,
                num_samples=29,
                seed=11,
                num_replicas=2,
                rank=rank,
            )
            for rank in range(2)
        ]
        shards = [list(sampler) for sampler in samplers]
        self.assertEqual(len(shards[0]), len(shards[1]))
        self.assertEqual(shards, [list(sampler) for sampler in samplers])

    def test_repository_wideresnet_supports_each_pair_space(self):
        """Exercise the repository model and its real ``sfc1`` parameters."""
        for dataset_name, (num_classes, num_superclasses) in DATASETS.items():
            image_sizes = (32, 64) if dataset_name == "smallimagenet" else (32,)
            for image_size in image_sizes:
                with self.subTest(dataset=dataset_name, image_size=image_size):
                    model = WideResNet(
                        num_classes=num_classes,
                        num_superclasses=num_superclasses,
                        depth=10,
                        widen_factor=1,
                    )
                    images = torch.randn(2, 3, image_size, image_size)
                    features = model(images)
                    targets_a = torch.tensor([0, 0])
                    targets_b = torch.tensor([1, num_superclasses - 1])
                    logits, pair_targets = mixed_superclass_logits(
                        features,
                        model.sfc1,
                        targets_a,
                        targets_b,
                        lam=0.5,
                    )
                    loss = F.cross_entropy(logits, pair_targets)
                    loss.backward()
                    self.assertEqual(
                        logits.shape[1],
                        num_superclasses * (num_superclasses - 1) // 2,
                    )
                    self.assertTrue(torch.isfinite(model.sfc1.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
