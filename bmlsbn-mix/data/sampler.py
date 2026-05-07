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
        

class BalancedMixedLabelSampler(torch.utils.data.sampler.Sampler):

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
        


class DualBalancedMixedLabelSampler(torch.utils.data.sampler.Sampler):

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


class DoubleBalancedMixedLabelSampler(torch.utils.data.sampler.Sampler):

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
        indices_c, indices_d = self.generate_pairwise_data()
        list_zip = list(zip(indices_a, indices_b, indices_c, indices_d))
        np.random.shuffle(list_zip)
        return iter(list_zip)
        
    def __len__(self):
        return self.num_samples

    def init_palette(self, num_samples, num_classes, num_class_list=None):
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
