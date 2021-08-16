#coding:utf-8
import numpy as np
import sys
import os

def bcubed(groundtruth, prediction, labeled_nodeid):
    gt_label_counts = dict(zip(*np.unique(groundtruth, return_counts=True)))
    pred_label_counts = dict(zip(*np.unique(prediction, return_counts=True)))

    recall = 0
    precision = 0

    for idx, gt_label in enumerate(groundtruth):
        if idx%100==0:
            print(idx)
        # print(gt_label)
        nodeid = labeled_nodeid[idx]
        # print(nodeid)
        pred_label = prediction[nodeid]
        gt_num = gt_label_counts[gt_label]
        pred_num = pred_label_counts[pred_label]

        nodeid_set = set(np.where(prediction == pred_label)[0])
        tp = 0
        for idx2, gt_label2 in enumerate(groundtruth):
            if labeled_nodeid[idx2] in nodeid_set and gt_label2 == gt_label:
                tp += 1
        precision += tp / pred_num
        recall += tp / gt_num

    total = len(groundtruth)
    precision /= total
    recall /= total
    f1 = 2 * precision * recall / ((recall + precision) or 1.0)
    return precision, recall, f1


if __name__ == "__main__":
    labelpath, outprefix = sys.argv[1], sys.argv[2]
    final_pred = np.load('%s/final_pred.npy'%outprefix)
    print(len(final_pred))
    print(max(final_pred))
    print("cluster num ", len(set(final_pred)))
    gt_label =  np.load(os.path.join(labelpath, 'new_label.npy')).reshape(-1).astype(np.int32)
    labeled_nodeid = np.load(os.path.join(labelpath, 'new_nodeid.npy')).astype(np.int32)
    p, r, f = bcubed(gt_label, final_pred, labeled_nodeid)
    print("prec {:.4f}, recall {:.4f}, f1 {:.4f}".format(p, r, f))

