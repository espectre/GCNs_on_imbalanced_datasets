###################################################################
# File Name: train.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Thu 06 Sep 2018 10:08:49 PM CST
###################################################################

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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

import model
from feeder.feeder import Feeder
from utils import to_numpy
from utils.meters import AverageMeter
from utils.serialization import load_checkpoint 
from utils.utils import bcubed
from utils.graph import graph_propagation, graph_propagation_soft, graph_propagation_naive
from sklearn.metrics import normalized_mutual_info_score, precision_score, recall_score


def parse_args():
    parser = argparse.ArgumentParser()
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--print_freq', default=40, type=int)

    # Optimization args
    parser.add_argument('--features', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--k-at-hop', type=int, nargs='+', default=[20,5])
    parser.add_argument('--active_connection', type=int, default=5)
    parser.add_argument('--rand_val', type=int, default=0)
    parser.add_argument('--prop_step', type=float, default=0.6)
    parser.add_argument('--max_size', type=int, default=900)
    parser.add_argument('--pool', type=str, default='avg')

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
    parser.add_argument("--enable_clustering",
                        action='store_true',
                        help="do clustering after inference")
    
    # Test args
    parser.add_argument('--checkpoint', type=str, metavar='PATH', default='./logs/logs/best.ckpt')

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


def main():
    args = parse_args()
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    node_world_size = int(os.environ['WORLD_SIZE'])
    ngpus_per_node = torch.cuda.device_count()

    logging.info("=> node_world_size [{}], gpus_per_node [{}]".format(
        node_world_size, ngpus_per_node
    ))
    world_size = ngpus_per_node * node_world_size
    logging.info ("=> total world size [{}]".format(world_size))
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,  world_size, args))


def main_worker(gpu, ngpus_per_node, world_size, args):
    args.gpu = gpu
    torch.cuda.set_device(gpu)
    node_rank = int(os.environ['RANK'])
    process_rank = node_rank * ngpus_per_node + gpu
    dist.init_process_group(backend="nccl", world_size=world_size, rank=process_rank)

    valset = Feeder(args.val_feat_path,
                    args.val_knn_graph_path,
                    args.val_label_path,
                    args.local_label_path,
                    args.mode,
                    -1,
                    args.seed,
                    args.k_at_hop,
                    args.active_connection,
                    args.rand_val,
                    train=False,
                    infer_only=True)

    val_sampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
            valset, batch_size=args.batch_size, sampler=val_sampler,
            num_workers=args.workers, shuffle=False, pin_memory=True)

    ckpt = load_checkpoint(args.checkpoint)
    net = model.gcn(args.features).cuda(gpu)
    net.load_state_dict(ckpt['state_dict'])

    knn_graph = valset.knn_graph
    knn_graph_dict = list() 
    for neighbors in knn_graph:
        knn_graph_dict.append(dict())
        for n in neighbors[1:]:
            knn_graph_dict[-1][n] = []

    criterion = nn.CrossEntropyLoss().cuda(gpu)
    net = DDP(net, device_ids=[gpu])

    edges, scores = validate(valloader, net, criterion, args, process_rank)

    torch.distributed.barrier()
    gather_edges = [torch.ones_like(edges) for _ in range(world_size)]
    gather_scores = [torch.ones_like(scores) for _ in range(world_size)]
    torch.distributed.all_gather(gather_edges, edges)
    torch.distributed.all_gather(gather_scores, scores)

    if process_rank == 0:

        gather_edges = np.array(torch.cat([x.cpu() for x in gather_edges], dim=0))
        gather_scores = np.array(torch.cat([x.cpu() for x in gather_scores], dim=0))

        prefix = os.path.dirname(args.checkpoint)
        np.save('%s/gather_edges'%prefix, gather_edges)
        np.save('%s/gather_scores'%prefix, gather_scores)

        print('Inference Completed !!!')

        if args.enable_clustering:
            # clusters = graph_propagation(gather_edges, gather_scores, max_sz=900, step=0.6, pool='avg')
            clusters = graph_propagation(gather_edges, gather_scores, max_sz=args.max_size, step=args.prop_step, pool=args.pool)
        
            final_pred = clusters2labels(clusters, len(valset))
            # labels = valset.labels
            # 20200604 add this line
            # labels = labels.reshape(-1).astype(np.int32)

            # np.save('%s/labels'%prefix, labels)
            np.save('%s/final_pred'%prefix, final_pred)
            print('Clustering Completed !!!')

            # TODO: Add a function to save cluster predicts and ground-truths as txt files


            # 20200614 add this for part test
            #left_label =  np.load(os.path.join(os.path.dirname(args.val_label_path), 'new_label.npy'))
            #left_nodeid = np.load(os.path.join(os.path.dirname(args.val_label_path), 'new_nodeid.npy'))
            #labels = left_label.reshape(-1).astype(np.int32)
            #final_pred = final_pred[left_nodeid]

            print('------------------------------------')
            print('Number of nodes: ', len(valset))
            print('Number of clusters: ', len(set(final_pred)))
            # print('Precision   Recall   F-Sore   NMI')
            # p,r,f = bcubed(labels, final_pred)
            # nmi = normalized_mutual_info_score(final_pred, labels)
            # print(('{:.4f}    '*4).format(p,r,f, nmi))

            # labels, final_pred = single_remove(labels, final_pred)
            # print('------------------------------------')
            # print('After removing singleton culsters, number of nodes: ', len(labels))
            # print('Number of clusters: ', len(set(final_pred)))
            # print('Precision   Recall   F-Sore   NMI')
            # p,r,f = bcubed(final_pred, labels)
            # nmi = normalized_mutual_info_score(final_pred, labels)
            # print(('{:.4f}    '*4).format(p,r,f, nmi))
        
    
def clusters2labels(clusters, n_nodes):
    labels = (-1)* np.ones((n_nodes,))
    for ci, c in enumerate(clusters):
        for xid in c:
            labels[xid.name] = ci
    assert np.sum(labels<0) < 1
    return labels 


# def labels2clusters(labels):
#     lb2idxs = {}
#     idx2lb = {}
#     label = np.load(fn_npy)
#     for idx, x in enumerate(label):
#         # lb = int(x.strip())
#         lb = x[0]
#         if lb not in lb2idxs:
#             lb2idxs[lb] = []
#         lb2idxs[lb] += [idx]
#         idx2lb[idx] = lb

#     inst_num = len(idx2lb)
#     cls_num = len(lb2idxs)
#     if verbose:
#         print('[{}] #cls: {}, #inst: {}'.format(fn_npy, cls_num, inst_num))
#     return lb2idxs, idx2lb, label


def make_labels(gtmat):
    return gtmat.view(-1)


def validate(loader, net, crit, args, process_rank):
    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    # losses = AverageMeter()
    # accs  = AverageMeter()
    # precisions  = AverageMeter()
    # recalls  = AverageMeter()
    
    net.eval()
    # end = time.time()
    edges = list()
    scores = list()
    for i, (feat, adj, cid, h1id, node_list) in enumerate(loader):
        # data_time.update(time.time() - end)
        feat, adj, cid, h1id = map(lambda x: x.cuda(args.gpu, non_blocking=True), 
                                (feat, adj, cid, h1id))
        pred = net(feat, adj, h1id)
        # labels = make_labels(gtmat).long()
        # loss = crit(pred, labels)
        pred = F.softmax(pred, dim=1)
        # p,r, acc = accuracy(pred, labels)
        
        # losses.update(loss.item(),feat.size(0))
        # accs.update(acc.item(),feat.size(0))
        # precisions.update(p, feat.size(0))
        # recalls.update(r,feat.size(0))
    
        # batch_time.update(time.time()- end)
        # end = time.time()
        if i % args.print_freq == 0 and process_rank == 0:
            print(i)
        #     print('[{0}/{1}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {losses.val:.3f} ({losses.avg:.3f})\n'
        #           'Accuracy {accs.val:.3f} ({accs.avg:.3f})\t'
        #           'Precison {precisions.val:.3f} ({precisions.avg:.3f})\t'
        #           'Recall {recalls.val:.3f} ({recalls.avg:.3f})'.format(
        #                 i, len(loader), batch_time=batch_time,
        #                 data_time=data_time, losses=losses, accs=accs, 
        #                 precisions=precisions, recalls=recalls))
        
        #print(node_list)
        node_list = node_list.long().squeeze(dim=1).numpy()
        bs = feat.size(0)
        for b in range(bs): 
            cidb = cid[b].int().item() 
            nl = node_list[b]

            for j,n in enumerate(h1id[b]):
                n = n.item()
                edges.append([nl[cidb], nl[n]])
                scores.append(pred[b*args.k_at_hop[0]+j,1].item())
    
    edges = torch.tensor(edges, dtype=torch.int32).cuda(args.gpu)
    scores = torch.tensor(scores, dtype=torch.float32).cuda(args.gpu)
    #edges = np.asarray(edges)
    #scores = np.asarray(scores)
    return edges, scores


def accuracy(pred, label):
    pred = torch.argmax(pred, dim=1).long()
    acc = torch.mean((pred == label).float())
    pred = to_numpy(pred)
    label = to_numpy(label)
    p = precision_score(label, pred)
    r = recall_score(label, pred)
    return p,r,acc 


if __name__ == '__main__':
    main()
