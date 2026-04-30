import torch

import numpy as np


def produce_Ew(label, num_classes):
    uni_label, count = torch.unique(label, return_counts=True)
    batch_size = label.size(0)
    uni_label_num = uni_label.size(0)
    assert batch_size == torch.sum(count)
    gamma = batch_size / uni_label_num
    Ew = torch.ones(1, num_classes).cuda(label.device)
    for i in range(uni_label_num):
        label_id = uni_label[i]
        label_count = count[i]
        length = torch.sqrt(gamma / label_count)
        Ew[0, label_id] = length
    return Ew


def dot_loss(features, targets, cur_M, H_length, reg_lam=0, type_='reg_dot_loss'):
    weight = cur_M[:, targets].T
    if type_ == 'dot_loss':
        loss = - torch.bmm(features.unsqueeze(1), weight.unsqueeze(2)).view(-1).mean()
    elif type_ == 'reg_dot_loss':
        dot = torch.bmm(features.unsqueeze(1), weight.unsqueeze(2)).view(-1)

        with torch.no_grad():
            M_length = torch.sqrt(torch.sum(weight ** 2, dim=1, keepdims=False))
        loss = 0.5 * torch.mean(((dot-(M_length * H_length)) ** 2) / H_length)

        if reg_lam > 0:
            reg_Eh_l2 = torch.mean(torch.sqrt(torch.sum(features ** 2, dim=1, keepdims=True)))
            loss = loss + reg_Eh_l2 * reg_lam

    return loss

