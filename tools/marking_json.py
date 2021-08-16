#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/21 5:14 下午
# @Author  : Enzo
# @File    : marking_json.py.py

import os
import time
import pickle
import numpy as np


def l2norm(vec):
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec


def main():
    with open(root_dir+pickle_file, 'rb') as f:
        d = pickle.load(f)
    d = d['data']
    print('total_num： ', len(d))

    feats = []
    face_ids = []
    no_fea_cnt = 0
    for i in range(len(d)):
        try:
            feat = list(map(np.float32, d[i][47][1:-1].split(',')))
            feats.append(feat)
            face_ids.append(d[i][1])
        except:
            no_fea_cnt += 1
    print('valuable num：', len(face_ids))
    print('no_feature num: ', no_fea_cnt)

    feats = l2norm(feats)
    if not os.path.exists('./features'):
        os.makedirs('./features')
    np.save('./features/feats.npy', feats)
    np.save('./features/face_ids.npy', face_ids)
    if len(face_ids) % 2 == 1:
        np.save('./features/feats_1.npy', feats[:-1])
        np.save('./features/face_ids_1.npy', face_ids[:-1])


if __name__ == '__main__':
    root_dir = './'
    pickle_file = '../20200811_pickle.pickle'
    main()




