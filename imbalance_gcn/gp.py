#coding:utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import os.path as osp
import sys
import time
import argparse
import numpy as np
import logging
import gc
from utils.utils import bcubed
from utils.graph import graph_propagation, graph_propagation_soft, graph_propagation_naive
from sklearn.metrics import normalized_mutual_info_score, precision_score, recall_score
from feeder.feeder import Feeder

def parse_args():
    parser = argparse.ArgumentParser()
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--workers', default=16, type=int)
    parser.add_argument('--print_freq', default=40, type=int)

    # Optimization args
    parser.add_argument('--features', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_sz', type=int, default=900)
    parser.add_argument('--step', type=float, default=0.1)
    parser.add_argument('--beg_th', type=float, default=0.6)
    parser.add_argument('--k-at-hop', type=int, nargs='+', default=[20,5])
    parser.add_argument('--active_connection', type=int, default=5)

    # Validation args
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--val_feat_path', type=str, metavar='PATH',
                        default=osp.join(working_dir, '../facedata/1024.fea.npy'))
    parser.add_argument('--val_knn_graph_path', type=str, metavar='PATH',
                        default=osp.join(working_dir, '../facedata/knn.graph.1024.bf.npy'))
    parser.add_argument('--val_label_path', type=str, metavar='PATH',
                        default=osp.join(working_dir, '../facedata/1024.labels.npy'))
    parser.add_argument('--local_label_path', type=str, default=None)
    parser.add_argument('--mode', type=str, default='sup')
    # Test args
    parser.add_argument('--checkpoint', type=str, metavar='PATH', default='./logs/logs/best.ckpt')
    parser.add_argument('--final_pred_size', type=int, default=900)

    args = parser.parse_args()
    return args

def single_remove(Y, pred):
    single_idcs = np.zeros_like(pred)
    pred_unique = np.unique(pred)
    for u in pred_unique:
        idcs = pred == u
        if np.sum(idcs) == 1:
            single_idcs[np.where(idcs)[0][0]] = 1
    remain_idcs = [i for i in range(len(pred)) if not single_idcs[i]]
    remain_idcs = np.asarray(remain_idcs)
    return Y[remain_idcs], pred[remain_idcs]

def clusters2labels(clusters, n_nodes):
    labels = (-1)* np.ones((n_nodes,))
    for ci, c in enumerate(clusters):
        for xid in c:
            labels[xid.name] = ci
    assert np.sum(labels<0) < 1
    return labels


def make_labels(gtmat):
    return gtmat.view(-1)

if __name__ == "__main__":
    args = parse_args()
    prefix = os.path.dirname(args.checkpoint)
    max_sz = args.max_sz
    step = args.step
    beg_th = args.beg_th
    print("max_sz: ", max_sz)
    print("step: ", step)
    print("beg_th: ", beg_th)

    gather_edges = np.load('%s/gather_edges.npy'%prefix)
    gather_scores = np.load('%s/gather_scores.npy'%prefix)
    # labels = np.load('%s/labels.npy'%prefix)
    # compatibale to the case where labels are not stored as a matrix of shape N
    # labels = labels.reshape(-1).astype(np.int32)

    # prediction file name without the ".npy" extension
    pred_fn = '%s/final_pred_'%prefix + str(max_sz) + '_' + str(beg_th) + '_' + str(step)
    if os.path.isfile(pred_fn + '.npy'):
        final_pred = np.load(pred_fn + '.npy')
        print(pred_fn+'.npy loaded')
    else:
        print("Working on graph propagation ...")
        clusters = graph_propagation(gather_edges, gather_scores, max_sz=max_sz, step=step, beg_th=beg_th, pool='avg')
        # final_pred = clusters2labels(clusters, len(labels))
        final_pred = clusters2labels(clusters, args.final_pred_size)
        np.save(pred_fn, final_pred)
        print(pred_fn+'.npy saved')

    # Release unreferenced memory
    gc.collect()

    # print('------------------------------------')
    # print('Number of nodes: ', len(labels))
    # print('Number of clusters: ', len(set(final_pred)))
    # print('Precision   Recall   F-Sore   NMI')
    # p,r,f = bcubed(labels, final_pred)
    # nmi = normalized_mutual_info_score(final_pred, labels)
    # print(('{:.4f}    '*4).format(p,r,f, nmi))
    #
    # labels, final_pred = single_remove(labels, final_pred)
    # print('------------------------------------')
    # print('After removing singleton culsters, number of nodes: ', len(labels))
    # print('Number of clusters: ', len(set(final_pred)))
    # print('Precision   Recall   F-Sore   NMI')
    # p,r,f = bcubed(final_pred, labels)
    # nmi = normalized_mutual_info_score(final_pred, labels)
    # print(('{:.4f}    '*4).format(p,r,f, nmi))

