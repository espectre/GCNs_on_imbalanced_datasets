#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/21 10:41 下午
# @Author  : Enzo
# @File    : losses.py.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# normal ce
def normal(crit, pred, labels):
    ce_loss = crit(pred, labels)
    return ce_loss


def weighted_ce(crit, pred, labels, weights):
    ce_loss = crit(pred, labels)
    weighted_ce_loss = ce_loss * weights
    return weighted_ce_loss


# ohem: select hard examples
def ohem(crit, pred, labels, ratio):
    ce_loss = crit(pred, labels)

    num_examples = labels.shape[0]

    # 根据个人喜好只选最难的一半
    n_selected = int(num_examples * ratio)
    vals, idxs = torch.topk(ce_loss, n_selected)

    th = vals[-1]
    selected_mask = ce_loss >= th
    loss_weight = selected_mask.float()
    return torch.sum(ce_loss * loss_weight) / torch.sum(loss_weight)


# ohem3: select neg:pos=3:1
def ohem3(crit, pred, labels):
    ce_loss = crit(pred, labels)

    pos_weight = torch.eq(labels, 1).float()
    n_pos = torch.sum(pos_weight)

    n_neg = torch.sum(1-pos_weight)
    
    # 防止选择数量超过negative sample的个数
    n_selected = torch.min(n_pos * 3, n_neg)
    
    # 防止出现什么也没选
    n_selected = torch.max(n_selected, torch.FloatTensor([1]).cuda())
    neg_mask = torch.eq(labels, 0)
    hardness = torch.where(neg_mask, ce_loss, torch.zeros_like(ce_loss))
    vals, idxs = torch.topk(hardness, n_selected.int().item())
    th = vals[-1]
    selected_neg_mask = torch.logical_and(hardness >= th, neg_mask)
    neg_weight = selected_neg_mask.float()

    loss_weight = pos_weight + neg_weight

    return torch.sum(ce_loss * loss_weight) / torch.sum((loss_weight))


# class_balance_ce
def class_balance_ce(crit, pred, labels):
    ce_loss = crit(pred, labels)
    pos_weight = torch.eq(labels, 1).float()
    neg_weight = 1 - pos_weight

    n_pos = torch.sum(pos_weight)
    n_neg = torch.sum(neg_weight)

    def has_pos():
        return torch.sum(ce_loss * pos_weight) / n_pos

    def has_neg():
        return torch.sum(ce_loss * neg_weight) / n_neg

    def no():
        return torch.tensor(0.0)

    if n_pos > 0:
        pos_loss = has_pos()
    else:
        pos_loss = no()
    if n_neg > 0:
        neg_loss = has_neg()
    else:
        neg_loss = no()

    return (pos_loss + neg_loss) / 2.0


class FocalLoss1(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 1e-3

    def forward(self, pred, label):
        assert pred.shape[0] == label.shape[0]
        assert pred.shape[1] == 2

        #assert label.shape[1] == 1

        # print(pred, label)
        # pred [N] is probability of positive,     label [N] \in {0, 1}
        log_prob = F.log_softmax(pred, dim=1)
        log_prob_pos = log_prob[:, 1]
        log_prob_neg = log_prob[:, 0]
        pos_term = - (1 - self.alpha) *  log_prob_neg.exp().detach() ** self.gamma * log_prob_pos * (label >= 1).float()
        #print('any', torch.isnan(pos_term).any(), flush=True)
        assert not torch.isnan(pos_term).any(), "pred {} alpha {} lpn {} g {} lpb {} l {}".format(pred, self.alpha, log_prob_neg, self.gamma, log_prob_pos, label)


        neg_term = - self.alpha * log_prob_pos.exp().detach() ** self.gamma * log_prob_neg * (label == 0).float()
        # print('any_neg', torch.isnan(neg_term).any(), flush=True)
        assert not torch.isnan(neg_term).any(), "{} {} {} {} {} {}".format(pred, self.alpha, log_prob_pos, self.gamma, log_prob_neg, label)

        # assert not torch.isnan(neg_term).any(), neg_term
        final = (pos_term + neg_term).mean()
        # assert final > 0
        assert not torch.isnan(final).any(), final
        return 2.0  * final

