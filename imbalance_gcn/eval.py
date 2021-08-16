from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics import normalized_mutual_info_score


def contingency_matrix(ref_labels, sys_labels):
    """Return contingency matrix between ``ref_labels`` and ``sys_labels``."""
    ref_classes, ref_class_inds = np.unique(ref_labels, return_inverse=True)
    sys_classes, sys_class_inds = np.unique(sys_labels, return_inverse=True)
    n_frames = ref_labels.size
    # Following works because coo_matrix sums duplicate entries. Is roughly
    # twice as fast as np.histogram2d.
    cmatrix = coo_matrix(
        (np.ones(n_frames), (ref_class_inds, sys_class_inds)),
        shape=(ref_classes.size, sys_classes.size),
        dtype=np.int)
    cmatrix = cmatrix.toarray()
    return cmatrix, ref_classes, sys_classes


def bcubed(ref_labels, sys_labels, cm=None):
    """Return B-cubed precision, recall, and F1.

    The B-cubed precision of an item is the proportion of items with its
    system label that share its reference label (Bagga and Baldwin, 1998).
    Similarly, the B-cubed recall of an item is the proportion of items
    with its reference label that share its system label. The overall B-cubed
    precision and recall, then, are the means of the precision and recall for
    each item.

    Parameters
    ----------
    ref_labels : ndarray, (n_frames,)
        Reference labels.

    sys_labels : ndarray, (n_frames,)
        System labels.

    cm : ndarray, (n_ref_classes, n_sys_classes)
        Contingency matrix between reference and system labelings. If None,
        will be computed automatically from ``ref_labels`` and ``sys_labels``.
        Otherwise, the given value will be used and ``ref_labels`` and
        ``sys_labels`` ignored.
        (Default: None)

    Returns
    -------
    precision : float
        B-cubed precision.

    recall : float
        B-cubed recall.

    f1 : float
        B-cubed F1.

    References
    ----------
    Bagga, A. and Baldwin, B. (1998). "Algorithms for scoring coreference
    chains." Proceedings of LREC 1998.
    """
    if cm is None:
        cm, _, _ = contingency_matrix(ref_labels, sys_labels)
    cm = cm.astype('float64')
    cm_norm = cm / cm.sum()
    precision = np.sum(cm_norm * (cm / cm.sum(axis=0)))
    recall = np.sum(cm_norm * (cm / np.expand_dims(cm.sum(axis=1), 1)))
    f1 = 2*(precision*recall)/(precision + recall)
    return precision, recall, f1

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

if __name__ == '__main__':
    label_path, pred_path = sys.argv[1], sys.argv[2]
    label = np.load(label_path)
    pred = np.load(pred_path)
    
    print('------------------------------------')
    print('Number of nodes: ', len(label))
    print('Number of clusters: ',len(set(pred)))
    print('Precision   Recall   F-Sore   NMI')
    p,r,f = bcubed(label, pred)
    nmi = normalized_mutual_info_score(pred, label)
    print(('{:.4f}    '*4).format(p,r,f, nmi))

    label, pred = single_remove(label, pred)
    print('------------------------------------')
    print('After removing singleton culsters, number of nodes: ', len(label))
    print('Number of clusters: ',len(set(pred)))
    print('Precision   Recall   F-Sore   NMI')
    p,r,f = bcubed(pred, label)
    nmi = normalized_mutual_info_score(pred, label)
    print(('{:.4f}    '*4).format(p,r,f, nmi))
    
