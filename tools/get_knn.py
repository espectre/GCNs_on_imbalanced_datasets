# coding:utf-8
# len of feats.npy needs even, instead of odd
import numpy as np
import faiss
from tqdm import tqdm
import sys
import time
import os
import argparse


nlist = 1000    # 1000 cluster for 100w ,选择的聚类中心数量
nprobe = 100    # test 10 cluster
# topk = 80
bs = 1000
# data_prefix = '/mnt/disk/gcn_cluster/20200811_clean/mark0820/features'
# # data_prefix = '/mnt/disk/face_cluster/celeb_ltc/features'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_prefix', type=str, metavar='PATH',
                        default='/workspace/data/semi_celeb/features')
    parser.add_argument('--feats_file', type=str, metavar='PATH',
                        default='./test_fea.npy')
    parser.add_argument('--topk', type=int, default=80)

    # parser.add_argument('--knn_file', type=str, metavar='PATH',
    #                     default='./knn.npy')
    args = parser.parse_args()
    return args


def batch_search(index, query, topk, bs, verbose=False):
    n = len(query)
    dists = np.zeros((n, topk), dtype=np.float32)
    nbrs = np.zeros((n, topk), dtype=np.int32)

    for sid in tqdm(range(0, n, bs), desc="faiss searching...", disable=not verbose):
        eid = min(n, sid + bs)
        dists[sid:eid], nbrs[sid:eid] = index.search(query[sid:eid], topk)
    cos_dist = dists / 2
    return cos_dist, nbrs

def get_knn():
    args = parse_args()
    featfile = os.path.join(args.data_prefix, args.feats_file)
    # featfile = os.path.join(data_prefix, 'day_cluster', 'feat.npy')

    query_arr = np.load(featfile, allow_pickle=True)
    # print(type(query_arr[0][0]))
    doc_arr = query_arr

    print("configure faiss")
    num_gpu = faiss.get_num_gpus()
    dim = query_arr.shape[1]
    quantizer = faiss.IndexFlatL2(dim)
    cpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist)
    cpu_index.nprobe = nprobe

    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    co.usePrecomputed = False
    co.shard = True
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co, ngpu=num_gpu)

    # start IVF
    print("build index")
    gpu_index.train(doc_arr)
    gpu_index.add(doc_arr)
    print(gpu_index.ntotal)

    # start query
    print("start query")
    gpu_index.nprobe = nprobe # default nprobe is 1, try a few more
    D, I = batch_search(gpu_index, query_arr, args.topk, bs, verbose=True)
    knn_path = args.feats_file.split('.')[0]
    if not os.path.exists(os.path.join(args.data_prefix, knn_path)):
        os.makedirs(os.path.join(args.data_prefix, knn_path))
    np.save(os.path.join(args.data_prefix, knn_path, 'knn_index_'+str(args.topk)), I)
    print("save knn index", I.shape)
    np.save(os.path.join(args.data_prefix, knn_path, 'knn_dist_'+str(args.topk)), D)
    data = np.concatenate((I[:,None,:], D[:,None,:]), axis=1)
    np.savez(os.path.join(args.data_prefix, knn_path, 'knn_data_'+str(args.topk)), data=data)
    # D:distance
    # I:index
    # data:npz

if __name__ == "__main__":
    beg_time = time.time()
    get_knn()
    print("time use %.4f"%(time.time()-beg_time))

