# This code is copied from the official PyTorch implementation in MiSLAS
# (https://github.com/dvlab-research/MiSLAS)
import math
import random
import numpy as np
from itertools import combinations, combinations_with_replacement, product

import torch
import torch.distributed as dist
from torch.utils.data.distributed import Sampler, Dataset

from typing import TypeVar, Optional, Iterator

T_co = TypeVar('T_co', covariant=True)

__all__ = [
    'BalancedDatasetSampler',
    'EffectNumSampler',
    'RandomCycleIter',

    'ClassAwareSampler',
    'BalancedMixedLabelSampler',
]


class BalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None, **kwargs):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = [0] * len(np.unique(dataset.targets))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1
        per_cls_weights = 1 / np.array(label_to_count)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        
        
        self.weights = torch.DoubleTensor(weights)
        
    def _get_label(self, dataset, idx):
        return dataset.targets[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples


class EffectNumSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None, **kwargs):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = [0] * len(np.unique(dataset.targets))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1

        

        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        
        
        self.weights = torch.DoubleTensor(weights)
        
    def _get_label(self, dataset, idx):
        return dataset.targets[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples


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


def class_aware_sample_generator(cls_iter, data_iter_list, n, num_samples_per_cls=1):

    i = 0
    j = 0
    while i < n:
        if j >= num_samples_per_cls:
            j = 0
    
        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]]*num_samples_per_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]
        
        i += 1
        j += 1

class ClassAwareSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, num_samples_per_cls=4, **kwargs):
        # pdb.set_trace()
        num_classes = len(np.unique(data_source.targets))
        self.class_iter = RandomCycleIter(range(num_classes))
        cls_data_list = [list() for _ in range(num_classes)]
        for i, label in enumerate(data_source.targets):
            cls_data_list[label].append(i)
        self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
        self.num_samples = max([len(x) for x in cls_data_list]) * len(cls_data_list)
        self.num_samples_per_cls = num_samples_per_cls
        
    def __iter__ (self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list,
                                            self.num_samples, self.num_samples_per_cls)
    
    def __len__ (self):
        return self.num_samples
 

def get_sampler():
    return ClassAwareSampler


class BalancedMixedLabelSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, cfg, dataset, num_classes=None, num_class_list=None, **kwargs):
        
        self.cfg = cfg
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
        self.palette_cp = self.palette

    def get_indices_iter(self, targets, num_classes):
        indices_per_cls = [list() for _ in range(num_classes)]
        for i, tgt in enumerate(targets):
            indices_per_cls[tgt].append(i)
        
        return [RandomCycleIter(idx) for idx in indices_per_cls]

    def distribute_remainder(self, palette, remainder):
        candidates = np.vstack(list(self.lbl_mix2new.keys()))
        np.random.shuffle(candidates)
        indices_cand = candidates[:remainder]
        palette_cp = np.copy(palette)
        palette_cp[indices_cand[:,0], indices_cand[:,1]] += 1

        self.palette_cp = palette_cp

    def generate_pairwise_data(self):
        indices_a, indices_b = [], []
        for mixed_lbl, new_lbl_num in self.lbl_mix2new.items():
            num_samples_per_mixed_cls = self.palette_cp[mixed_lbl]
            for _ in range(num_samples_per_mixed_cls):
                indices_a.append(next(self.indices_iter[mixed_lbl[0]]))
                indices_b.append(next(self.indices_iter[mixed_lbl[1]]))

        return indices_a, indices_b

