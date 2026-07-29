# This code is copied from the official PyTorch implementation in MiSLAS
# (https://github.com/dvlab-research/MiSLAS)
import math
import random
import numpy as np
from itertools import combinations, combinations_with_replacement

import torch
import torch.distributed as dist
from torch.utils.data.distributed import Sampler, Dataset

from typing import TypeVar, Optional, Iterator

T_co = TypeVar('T_co', covariant=True)

__all__ = [
    'MixedLabelDataset',
    'BalancedMixedLabelSampler',
    'DualMixedLabelDataset',
    'DualBalancedMixedLabelSampler',
    'DoubleMixedLabelDataset',
    'DoubleBalancedMixedLabelSampler',
]


class _ScalablePairSampler(object):
    """Memory-efficient fallback over all ordered class pairs.

    A cycle contains every ordered pair, including same-class pairs, exactly
    once. It is represented as randomized permutation matchings instead of a
    dense C-by-C matrix. Every consecutive block of C pairs contains each
    class exactly once on the left and right. The existing dense behavior is
    retained when an epoch is large enough to cover its complete pair space.
    """

    def _maybe_init_sparse(self, num_samples, num_classes, num_class_list, allow_same):
        # Retain ``allow_same`` in the signature for compatibility. Circular
        # BMLS intentionally includes diagonal pairs and treats (a, b) and
        # (b, a) as distinct mixed classes.
        del allow_same
        if num_class_list is None:
            available_classes = sorted(set(int(target) for target in self.dataset.targets))
        else:
            available_classes = [
                class_index
                for class_index, count in enumerate(num_class_list)
                if count > 0
            ]
        if not available_classes:
            raise ValueError("Pair sampling requires a non-empty class")

        num_available = len(available_classes)
        if num_samples >= num_available ** 2:
            self.sparse_mode = False
            return False

        # Keep the old attribute name because the sampler implementations use
        # it to select their non-dense path.
        self.sparse_mode = True
        self.available_classes = np.asarray(available_classes, dtype=np.int64)
        self.palette = None
        self.remainder = num_samples
        first = int(self.available_classes[0])
        self.lbl_mix2new = {(first, first): 0}
        self._circular_epoch = 0
        self._circular_stream = 0
        self._circular_streams_per_epoch = 1
        self._pair_schedule_ready = False
        self._circular_seed = int(
            getattr(getattr(self.dataset, 'args', None), 'seed', 0) or 0
        )
        self.palette_cp = 'circular ordered-pair schedule (not generated)'
        return True

    def set_epoch(self, epoch):
        """Select a deterministic, resume-safe segment of the pair cycle."""
        self._circular_epoch = int(epoch)
        self._circular_stream = 0
        self._pair_schedule_ready = False

    def _cycle_permutations(self, cycle_index):
        num_available = len(self.available_classes)
        seed_sequence = np.random.SeedSequence(
            [self._circular_seed, int(cycle_index)]
        )
        generator = np.random.default_rng(seed_sequence)
        return (
            generator.permutation(num_available),
            generator.permutation(num_available),
            generator.permutation(num_available),
        )

    def _generate_sparse_pairs(self):
        num_available = len(self.available_classes)
        cycle_size = num_available ** 2
        stream = self._circular_stream
        global_start = (
            self._circular_epoch
            * self._circular_streams_per_epoch
            * self.num_samples
            + stream * self.num_samples
        )
        left = np.empty(self.num_samples, dtype=np.int64)
        right = np.empty(self.num_samples, dtype=np.int64)

        written = 0
        position = global_start
        while written < self.num_samples:
            cycle_index, cycle_offset = divmod(position, cycle_size)
            take = min(self.num_samples - written, cycle_size - cycle_offset)
            row_order, column_order, shift_order = self._cycle_permutations(
                cycle_index
            )

            offsets = cycle_offset + np.arange(take, dtype=np.int64)
            matching_index = offsets // num_available
            row_position = offsets % num_available
            shifts = shift_order[matching_index]

            output_slice = slice(written, written + take)
            left[output_slice] = self.available_classes[
                row_order[row_position]
            ]
            right[output_slice] = self.available_classes[
                column_order[(row_position + shifts) % num_available]
            ]
            written += take
            position += take

        self._sparse_pairs = np.column_stack((left, right))
        self._pair_schedule_ready = True
        self._circular_stream += 1
        self.palette_cp = (
            'circular ordered-pair schedule '
            '(epoch={}, stream={}, start={}, samples={}, classes={})'
        ).format(
            self._circular_epoch,
            stream,
            global_start,
            self.num_samples,
            num_available,
        )

    def _generate_sparse_indices(self):
        if not self._pair_schedule_ready:
            self._generate_sparse_pairs()
        indices_a = []
        indices_b = []
        for class_a, class_b in self._sparse_pairs:
            indices_a.append(next(self.indices_iter[int(class_a)]))
            indices_b.append(next(self.indices_iter[int(class_b)]))
        self._pair_schedule_ready = False
        return indices_a, indices_b

        
class MixedLabelDataset(Dataset):
    def __init__(self, dataset):
        super(MixedLabelDataset, self).__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        #return [img_w, img_s], target, [None, None], None, sample_index_a, -1
        img, target, _, _, idx, _ = self.dataset.__getitem__(index[0])
        img_b, target_b, _, _, idx_b, _ = self.dataset.__getitem__(index[1])
        return img[1], img_b[1], target, target_b

    def __len__(self):
        return len(self.dataset)
        

class BalancedMixedLabelSampler(
        _ScalablePairSampler, torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, num_classes=None, num_class_list=None, **kwargs):
        self.dataset = dataset
        self.num_samples = len(dataset)
        self.num_classes = num_classes

        self.init_palette(self.num_samples, self.num_classes, num_class_list=num_class_list)
        self.indices_iter = self.get_indices_iter(dataset.targets, self.num_classes)
        
    def _get_label(self, dataset, idx):
        return dataset.targets[idx]
                
    def __iter__(self):
        indices_a, indices_b = self.generate_pairwise_data()
        list_zip = list(zip(indices_a, indices_b))
        np.random.shuffle(list_zip)
        return iter(list_zip)
        
    def __len__(self):
        return self.num_samples

    def init_palette(self, num_samples, num_classes, num_class_list=None):
        if self._maybe_init_sparse(
                num_samples, num_classes, num_class_list, allow_same=False):
            return
        n_cls = num_classes
        num_mixed_cls = n_cls*(n_cls-1)/2
         
        avg_num_samples_per_mixed_cls = num_samples // num_mixed_cls
        remainder = int(num_samples - avg_num_samples_per_mixed_cls * num_mixed_cls)
        
        palette = np.triu(torch.ones((num_classes, num_classes)), k=1) * avg_num_samples_per_mixed_cls
        lbl_mix2new = {tuple(v): i for i, v in enumerate(np.vstack(np.where(palette > 0)).T)}
        if len(lbl_mix2new) == 0:
            seq = [cls_num for cls_num in range(num_classes)]
            lbl_mix2new = {tuple(v): i for i, v in enumerate(combinations(seq, 2))}

        self.palette = palette.astype(np.int32)
        self.remainder = remainder
        self.lbl_mix2new = lbl_mix2new

    def get_palette(self):
        return self.palette, self.remainder, self.lbl_mix2new

    def set_palette_cp(self):
        if self.sparse_mode:
            return
        self.palette_cp = self.palette

    def get_indices_iter(self, targets, num_classes):
        indices_per_cls = [list() for _ in range(num_classes)]
        for i, tgt in enumerate(targets):
            indices_per_cls[tgt].append(i)
        
        return [RandomCycleIter(idx) for idx in indices_per_cls]

    def distribute_remainder(self, palette, remainder):
        if self.sparse_mode:
            self._generate_sparse_pairs()
            return
        candidates = np.vstack(list(self.lbl_mix2new.keys()))
        np.random.shuffle(candidates)
        indices_cand = candidates[:remainder]
        palette_cp = np.copy(palette)
        palette_cp[indices_cand[:,0], indices_cand[:,1]] += 1

        self.palette_cp = palette_cp

    def generate_pairwise_data(self):
        if self.sparse_mode:
            return self._generate_sparse_indices()
        indices_a, indices_b = [], []
        for mixed_lbl, new_lbl_num in self.lbl_mix2new.items():
            num_samples_per_mixed_cls = self.palette_cp[mixed_lbl]
            for _ in range(num_samples_per_mixed_cls):
                indices_a.append(next(self.indices_iter[mixed_lbl[0]]))
                indices_b.append(next(self.indices_iter[mixed_lbl[1]]))

        return indices_a, indices_b

class RandomCycleIter:

    def __init__ (self, data, test_mode=False):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode
        
    def __iter__ (self):
        return self
    
    def __next__ (self):
        self.i += 1
        
        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)
            
        return self.data_list[self.i]


class DualMixedLabelDataset(Dataset):
    def __init__(self, dataset):
        super(DualMixedLabelDataset, self).__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        #return [img_w, img_s], target, [None, None], None, sample_index_a, -1
        img, target, _, _, idx, _ = self.dataset.__getitem__(index[0])
        img_a, target_a, _, _, idx_a, _ = self.dataset.__getitem__(index[1])
        img_b, target_b, _, _, idx_b, _ = self.dataset.__getitem__(index[2])
        return img[1], img_a[1], img_b[1], target, target_a, target_b

    def __len__(self):
        return len(self.dataset)
        


class DualBalancedMixedLabelSampler(
        _ScalablePairSampler, torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, num_classes=None, num_class_list=None, **kwargs):
        self.dataset = dataset
        self.num_samples = len(dataset)
        self.num_classes = num_classes

        self.init_palette(self.num_samples, self.num_classes, num_class_list=num_class_list)
        self.indices_iter = self.get_indices_iter(dataset.targets, self.num_classes)
        
    def _get_label(self, dataset, idx):
        return dataset.targets[idx]
                
    def __iter__(self):
        indices_a, indices_b = self.generate_pairwise_data()
        indices = np.random.permutation(np.arange(self.num_samples))
        list_zip = list(zip(indices, indices_a, indices_b))
        np.random.shuffle(list_zip)
        return iter(list_zip)
        
    def __len__(self):
        return self.num_samples

    def init_palette(self, num_samples, num_classes, num_class_list=None):
        if self._maybe_init_sparse(
                num_samples, num_classes, num_class_list, allow_same=False):
            return
        n_cls = num_classes
        num_mixed_cls = n_cls*(n_cls-1)/2
         
        avg_num_samples_per_mixed_cls = num_samples // num_mixed_cls
        remainder = int(num_samples - avg_num_samples_per_mixed_cls * num_mixed_cls)
        
        palette = np.triu(torch.ones((num_classes, num_classes)), k=1) * avg_num_samples_per_mixed_cls
        lbl_mix2new = {tuple(v): i for i, v in enumerate(np.vstack(np.where(palette > 0)).T)}
        if len(lbl_mix2new) == 0:
            seq = [cls_num for cls_num in range(num_classes)]
            lbl_mix2new = {tuple(v): i for i, v in enumerate(combinations(seq, 2))}

        self.palette = palette.astype(np.int32)
        self.remainder = remainder
        self.lbl_mix2new = lbl_mix2new

    def get_palette(self):
        return self.palette, self.remainder, self.lbl_mix2new

    def set_palette_cp(self):
        if self.sparse_mode:
            return
        self.palette_cp = self.palette

    def get_indices_iter(self, targets, num_classes):
        indices_per_cls = [list() for _ in range(num_classes)]
        for i, tgt in enumerate(targets):
            indices_per_cls[tgt].append(i)
        
        return [RandomCycleIter(idx) for idx in indices_per_cls]

    def distribute_remainder(self, palette, remainder):
        if self.sparse_mode:
            self._generate_sparse_pairs()
            return
        candidates = np.vstack(list(self.lbl_mix2new.keys()))
        np.random.shuffle(candidates)
        indices_cand = candidates[:remainder]
        palette_cp = np.copy(palette)
        palette_cp[indices_cand[:,0], indices_cand[:,1]] += 1

        self.palette_cp = palette_cp

    def generate_pairwise_data(self):
        if self.sparse_mode:
            return self._generate_sparse_indices()
        indices_a, indices_b = [], []
        for mixed_lbl, new_lbl_num in self.lbl_mix2new.items():
            num_samples_per_mixed_cls = self.palette_cp[mixed_lbl]
            for _ in range(num_samples_per_mixed_cls):
                indices_a.append(next(self.indices_iter[mixed_lbl[0]]))
                indices_b.append(next(self.indices_iter[mixed_lbl[1]]))

        return indices_a, indices_b


class DoubleMixedLabelDataset(Dataset):
    def __init__(self, dataset):
        super(DoubleMixedLabelDataset, self).__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        #return [img_w, img_s], target, [None, None], None, sample_index_a, -1
        img_a, target_a, _, _, idx_a, _ = self.dataset.__getitem__(index[0])
        img_b, target_b, _, _, idx_b, _ = self.dataset.__getitem__(index[1])
        img_c, target_c, _, _, idx_c, _ = self.dataset.__getitem__(index[2])
        img_d, target_d, _, _, idx_d, _ = self.dataset.__getitem__(index[3])
        return img_a[1], img_b[1], img_c[1], img_d[1], target_a, target_b, target_c, target_d

    def __len__(self):
        return len(self.dataset)


class DoubleBalancedMixedLabelSampler(
        _ScalablePairSampler, torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, num_classes=None, num_class_list=None, **kwargs):
        self.dataset = dataset
        self.num_samples = len(dataset)
        self.num_classes = num_classes

        self.init_palette(self.num_samples, self.num_classes, num_class_list=num_class_list)
        self._circular_streams_per_epoch = 2
        self.indices_iter = self.get_indices_iter(dataset.targets, self.num_classes)
        
    def _get_label(self, dataset, idx):
        return dataset.targets[idx]
                
    def __iter__(self):
        indices_a, indices_b = self.generate_pairwise_data()
        indices_c, indices_d = self.generate_pairwise_data()
        list_zip = list(zip(indices_a, indices_b, indices_c, indices_d))
        np.random.shuffle(list_zip)
        return iter(list_zip)
        
    def __len__(self):
        return self.num_samples

    def init_palette(self, num_samples, num_classes, num_class_list=None):
        if self._maybe_init_sparse(
                num_samples, num_classes, num_class_list, allow_same=True):
            return
        n_cls = num_classes
        #num_mixed_cls = n_cls*(n_cls-1)/2
        num_mixed_cls = n_cls ** 2
         
        avg_num_samples_per_mixed_cls = num_samples // num_mixed_cls
        remainder = int(num_samples - avg_num_samples_per_mixed_cls * num_mixed_cls)
        
        #palette = np.triu(torch.ones((num_classes, num_classes)), k=1) * avg_num_samples_per_mixed_cls
        palette = np.ones((num_classes, num_classes)) * avg_num_samples_per_mixed_cls 
        lbl_mix2new = {tuple(v): i for i, v in enumerate(np.vstack(np.where(palette > 0)).T)}
        if len(lbl_mix2new) == 0:
            seq = [cls_num for cls_num in range(num_classes)]
            lbl_mix2new = {tuple(v): i for i, v in enumerate(combinations(seq, 2))}

        self.palette = palette.astype(np.int32)
        self.remainder = remainder
        self.lbl_mix2new = lbl_mix2new

    def get_palette(self):
        return self.palette, self.remainder, self.lbl_mix2new

    def set_palette_cp(self):
        if self.sparse_mode:
            return
        self.palette_cp = self.palette

    def get_indices_iter(self, targets, num_classes):
        indices_per_cls = [list() for _ in range(num_classes)]
        for i, tgt in enumerate(targets):
            indices_per_cls[tgt].append(i)
        
        return [RandomCycleIter(idx) for idx in indices_per_cls]

    def distribute_remainder(self, palette, remainder):
        if self.sparse_mode:
            self._generate_sparse_pairs()
            return
        candidates = np.vstack(list(self.lbl_mix2new.keys()))
        np.random.shuffle(candidates)
        indices_cand = candidates[:remainder]
        palette_cp = np.copy(palette)
        palette_cp[indices_cand[:,0], indices_cand[:,1]] += 1

        self.palette_cp = palette_cp

    def generate_pairwise_data(self):
        if self.sparse_mode:
            return self._generate_sparse_indices()
        indices_a, indices_b = [], []
        for mixed_lbl, new_lbl_num in self.lbl_mix2new.items():
            num_samples_per_mixed_cls = self.palette_cp[mixed_lbl]
            for _ in range(num_samples_per_mixed_cls):
                indices_a.append(next(self.indices_iter[mixed_lbl[0]]))
                indices_b.append(next(self.indices_iter[mixed_lbl[1]]))

        return indices_a, indices_b
