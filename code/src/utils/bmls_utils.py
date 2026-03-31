import torch

import numpy as np


def convert_to_singleton(tgts_a, tgts_b, lbl_mix2new, rank=None):
    new_tgts = []
    for tgt_a, tgt_b in zip(tgts_a, tgts_b):
        tgt_a, tgt_b = tgt_a.detach().data.item(), tgt_b.detach().data.item()
        mixed_tgt = (tgt_a, tgt_b) if tgt_a < tgt_b else (tgt_b, tgt_a)
        new_tgts.append(lbl_mix2new[mixed_tgt])
    new_tgts = torch.tensor(new_tgts, dtype=torch.int64)

    if rank is not None:
        new_tgts = new_tgts.cuda(rank)

    return new_tgts


def get_uni_and_pred_targets(targets):
    if isinstance(targets, tuple):
        tgts_a, tgts_b = targets[0], targets[1]
        uni_tgts = torch.unique(torch.cat([tgts_a, tgts_b]))
        pred_tgts_a = torch.where(tgts_a.reshape(-1, 1) == uni_tgts)[1]
        pred_tgts_b = torch.where(tgts_b.reshape(-1, 1) == uni_tgts)[1]
        return uni_tgts, (pred_tgts_a, pred_tgts_b)
    else:
        uni_tgts = torch.unique(targets)
        pred_tgts = torch.where(targets.reshape(-1, 1) == uni_tgts)[1]
        return uni_tgts, pred_tgts
    

def get_color(y_a, y_b, lam, num_classes=3, rank=None, class_check=False):
    batch_size = y_a.size()[0]

    n_cls = 2 if class_check else num_classes

    if rank is not None:
        colors = torch.zeros((batch_size, n_cls)).cuda(rank)
    else:
        colors = torch.zeros((batch_size, n_cls))

    for i in range(batch_size):
        if class_check:
            if y_a[i] == y_b[i]:
                colors[i][0] = 2 * abs(lam - 0.5)
            else:
                colors[i][1] = 2 * abs(lam - 0.5)

        else:
            colors[i][y_a[i]] += lam
            colors[i][y_b[i]] += (1 - lam)

    return colors

