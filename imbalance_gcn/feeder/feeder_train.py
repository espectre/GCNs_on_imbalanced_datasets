#coding:utf-8
###################################################################
# File Name: feeder.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Thu 06 Sep 2018 01:06:16 PM CST
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from math import ceil
import random
import torch
import torch.utils.data as data
import lmdb
import pickle
import os


def get_feat(env, key):
    with env.begin(write=False) as txn:
        value = txn.get(str(key).encode('utf-8'))
    feat = pickle.loads(value)
    return feat


class Feeder(data.Dataset):
    '''
    Generate a sub-graph from the feature graph centered at some node, 
    and now the sub-graph has a fixed depth, i.e. 2
    '''
    def __init__(self, feat_path, knn_graph_path, label_path, local_label_path, density_path, mode='sup', lambda_u=10,
                 seed=1, k_at_hop=[200, 5], active_connection=5, rand_val=0, rand_twice=0, resample=0, is_density=0,
                 clique_size=0, train=True, infer_only=False, loss='normal'):
        self.mode = mode
        self.train = train
        self.infer_only = infer_only
        self.is_density = is_density
        self.clique_size = clique_size
        self.lambda_u = lambda_u
        np.random.seed(seed)
        random.seed(seed)
        self.feat_path = feat_path
        self.knn_graph = np.load(knn_graph_path)
        self.loss = loss
        if self.loss == 'Weighted_CE':
            self.density = np.load(density_path)
            print(self.density.shape)

        if self.train and self.mode != 'sup':
            self.local_label = np.load(local_label_path)
            if self.mode == 'semi':
                self.labeled_node_list = np.load(os.path.join(os.path.dirname(label_path), 'label_node_list_200.npy')).astype(np.int32)
                self.unlabel_node_list = list(set(list(range(len(self.knn_graph)))) - set(self.labeled_node_list))
                print("load %s labeled node, %s unlabel node"%(len(self.labeled_node_list), len(self.unlabel_node_list)))
        elif not self.infer_only:
            self.labels = np.load(label_path)

        self.num_samples = len(self.knn_graph)
        self.depth = len(k_at_hop)
        self.k_at_hop = k_at_hop
        self.active_connection = active_connection
        self.rand_val = rand_val
        self.rand_twice = rand_twice
        self.resample = resample

        assert np.mean(k_at_hop) >= active_connection

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        '''
        return the vertex feature and the adjacent matrix A, together 
        with the indices of the center node and its 1-hop nodes
        '''
        if self.mode == 'semi':
            if index % self.lambda_u == 0:
                # choose a labeled nodeid
                index = np.random.choice(self.labeled_node_list)
            else:
                # choose a unlabel nodeid
                index = np.random.choice(self.unlabel_node_list)

        # hops[0] for 1-hop neighbors, hops[1] for 2-hop neighbors
        hops = list()
        center_node = index

        # new resample && random replace
        if self.resample or self.rand_val:
            cur_knn = self.knn_graph[center_node]
            if self.resample:
                cur_knn = cur_knn[:int(self.k_at_hop[0]*self.resample)]
            elif self.rand_val:
                cur_knn = cur_knn[:int(self.k_at_hop[0]*self.rand_val)]
            # res_idx = cur_knn[:self.k_at_hop[0]]
            cur_label = self.labels[cur_knn]
            pos_idx = cur_knn[cur_label == cur_label[0]]
            neg_idx = cur_knn[cur_label != cur_label[0]]
            pos_idx = pos_idx[1:]
            res_idx = None
            if len(pos_idx) == 0:
                res_idx = neg_idx[:self.k_at_hop[0]]
            elif len(neg_idx) == 0:
                res_idx = pos_idx[:self.k_at_hop[0]]
            else:
                # resample
                if self.resample:
                    if len(pos_idx) < self.k_at_hop[0]/2:
                        rate = self.k_at_hop[0]/2/len(pos_idx)
                        pos_idx = np.array(list(pos_idx) * ceil(rate))[:int(self.k_at_hop[0]/2)]
                        res_idx = np.array(list(pos_idx) + list(neg_idx[:ceil(self.k_at_hop[0]/2)]))
                    elif len(neg_idx) < self.k_at_hop[0]/2:
                        rate = self.k_at_hop[0]/2/len(neg_idx)
                        neg_idx = np.array(list(neg_idx) * ceil(rate))[:int(self.k_at_hop[0]/2)]
                        res_idx = np.array(list(neg_idx) + list(pos_idx[:ceil(self.k_at_hop[0]/2)]))
                    else:
                        res_idx = np.array(list(pos_idx[:ceil(self.k_at_hop[0]/2)]) + list(neg_idx[:int(self.k_at_hop[0]/2)]))
                # random
                elif self.rand_val:
                    pos_prob = (1 / 2 * 1 / len(pos_idx)) * np.ones(len(pos_idx))
                    neg_prob = (1 / 2 * 1 / len(neg_idx)) * np.ones(len(neg_idx))
                    total_idx = list(pos_idx) + list(neg_idx)
                    total_prob = list(pos_prob) + list(neg_prob)
                    res_idx = sorted(np.random.choice(total_idx, self.k_at_hop[0], p=total_prob, replace=False))
            hops.append(res_idx)
        else:
            hops.append(list(self.knn_graph[center_node][1:self.k_at_hop[0] + 1]))
            # hops.append(set(self.knn_graph[center_node][1:self.k_at_hop[0] + 1]))

        # hop1 no-set, hop2 set
        for d in range(1, self.depth):
            hops.append(set())
            for h in hops[-2]:
                # hops[-1] += list(self.knn_graph[h][1:self.k_at_hop[d] + 1])
                hops[-1].update(set(self.knn_graph[h][1:self.k_at_hop[d] + 1]))
        # 至此 hops 是 二维的 list

        hops_set = [h for hop in hops for h in hop]  # 结构shape 拆散成 1D
        hops_set.append(center_node)  # hops 里面是 global id

        max_num_nodes = self.k_at_hop[0] * (self.k_at_hop[1] + 1) + 1

        unique_nodes_list = list(hops_set)
        unique_nodes_map = {j:i for i,j in enumerate(unique_nodes_list)}  # 构建local id

        center_idx = torch.Tensor([unique_nodes_map[center_node],]).type(torch.long)
        one_hop_idcs = torch.Tensor([unique_nodes_map[i] for i in hops[0]]).type(torch.long)
        if self.loss == 'Weighted_CE':
            one_hop_density = self.density[one_hop_idcs]

        # 20200605 add this
        env = lmdb.open(self.feat_path, readonly=True, map_size=int(1e12))
        # center_feat = torch.Tensor(self.features[center_node]).type(torch.float)
        center_feat = torch.Tensor(get_feat(env, center_node)).type(torch.float)
        feat_list = []
        for each_node in unique_nodes_list:
            each_feat = get_feat(env, each_node)
            feat_list.append(each_feat)
        feat_arr = np.array(feat_list)
        feat = torch.Tensor(feat_arr).type(torch.float)
        # feat = torch.Tensor(self.features[unique_nodes_list]).type(torch.float)
        env.close()

        feat = feat - center_feat
        num_nodes = len(unique_nodes_list)
        A = torch.zeros(num_nodes, num_nodes)

        _, fdim = feat.shape
        feat = torch.cat([feat, torch.zeros(max_num_nodes - num_nodes, fdim)], dim=0)
      
        for node in unique_nodes_list:
            neighbors = self.knn_graph[node, 1:self.active_connection+1]
            for n in neighbors:
                if n in unique_nodes_list: 
                    A[unique_nodes_map[node], unique_nodes_map[n]] = 1
                    A[unique_nodes_map[n], unique_nodes_map[node]] = 1

        D = A.sum(1, keepdim=True)
        D = torch.max(D, torch.ones(D.size()))
        A = A.div(D)
        A_ = torch.zeros(max_num_nodes,max_num_nodes)
        A_[:num_nodes,:num_nodes] = A

        # 20200605 add this process
        if self.mode != 'sup':
            nodeid2locallabel = {}
            # if self.rand_val == 0 or is_random <= self.rand_twice:
            if self.resample == 0 and self.rand_val == 0:
                for idx, nodeid in enumerate(self.knn_graph[center_node][1:self.k_at_hop[0]+1]):
                    nodeid2locallabel[nodeid] = self.local_label[center_node][1:self.k_at_hop[0]+1][idx]
            else:
                # random
                for idx, nodeid in enumerate(res_idx):
                    nodeid2locallabel[nodeid] = self.local_label[center_node][1:][idx]

            # # thresh
            # for idx, nodeid in enumerate(self.knn_graph[center_node][1:thresh_idx]):
            #     nodeid2locallabel[nodeid] = self.local_label[center_node][1:thresh_idx][idx]


                # edge_labels = np.zeros(len(self.local_label[index][1:]), dtype=np.int32)
                ##label_list = [nodeid2locallabel[nodeid] for nodeid in unique_nodes_list if nodeid in hops[0]]
            label_list = [nodeid2locallabel[nodeid] for nodeid in hops[0]]
            edge_labels = torch.from_numpy(np.array(label_list, dtype=np.int32)).long()

        elif not self.infer_only:
            labels = self.labels[np.asarray(unique_nodes_list)]
            labels = torch.from_numpy(labels).type(torch.long)
            one_hop_labels = labels[one_hop_idcs]
            center_label = labels[center_idx]
            edge_labels = (center_label == one_hop_labels).long()

        if self.train:
            if self.loss == 'Weighted_CE':
                return (feat, A_, center_idx, one_hop_idcs), one_hop_density, edge_labels
            else:
                return (feat, A_, center_idx, one_hop_idcs), center_idx, edge_labels

        # Testing
        unique_nodes_list = torch.Tensor(unique_nodes_list)
        unique_nodes_list = torch.cat(
                [unique_nodes_list, torch.zeros(max_num_nodes-num_nodes)], dim=0)

        if self.infer_only:
            return(feat, A_, center_idx, one_hop_idcs, unique_nodes_list)
        else:
            return(feat, A_, center_idx, one_hop_idcs, unique_nodes_list), edge_labels


