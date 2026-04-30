import torch

import math
import numpy as np
from collections import Counter
from itertools import combinations

from dataset.sampler import RandomCycleIter


def convert_to_numpy(y):
    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy()
    elif isinstance(y, list):
        y_np = np.array(y)
    else:
        y_np = y

    return y_np


def get_index_randomly(batch_size, rank=None):
    if rank is not None:
        index = torch.randperm(batch_size).cuda(rank)
    else:
        index = torch.randperm(batch_size)
    
    return index


def pair_data(data, targets, pair_type=None, cnt_map=None, rank=None):
    if pair_type == 'random':
        assert data.shape[0] == len(targets)
        batch_size = len(targets)
        index = get_index_randomly(batch_size)
        return data, data[index], targets, targets[index]
    else:
        return data[0], data[1], targets[0], targets[1]


def mixup_data(x_a, x_b, alpha=None, lam=None):
    if lam is None:
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    mixed_x = lam * x_a + (1 - lam) * x_b
    return mixed_x, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_pred_targets(tgts_a, tgts_b, lam):
    if isinstance(lam, torch.Tensor):
        tgts = torch.hstack([tgts_a.reshape(-1, 1), tgts_b.reshape(-1, 1)])
        return tgts[torch.arange(len(lam)), (lam.squeeze() < 0.5).long()]
    else:
        return tgts_a if lam > 0.5 else tgts_b
            

def get_lam_pair(lam, type_='linear'):
    v_lam = 0.5 if lam is None else lam
    if type_ == 'etf':
        pi = 3.141592
        v_lam = np.sin(v_lam * pi/2)
        v_lam_a, v_lam_b = v_lam, np.sqrt(1 - v_lam ** 2)
    else:
        v_lam_a, v_lam_b = v_lam, 1 - v_lam

    return v_lam_a, v_lam_b

