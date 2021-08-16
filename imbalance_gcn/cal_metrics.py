import os
import numpy as np
from utils.utils import bcubed
from collections import Counter, defaultdict

# parameters
val_log = open('val_log_deploy.txt', 'a')
pred_dir = '/home/huafeng.yhf/cal_f_score_deploy'
final_pred = np.load(os.path.join(pred_dir, 'final_pred_v1_deploy_filt.npy'))
labels = np.load(os.path.join(pred_dir, 'labels_v1_deploy_filt.npy'))
cluster_size_list = [1, 2, 4]

for cluster_size in cluster_size_list:
    # cluster_size = 2
    val_log.write("Current cluster size is " + str(cluster_size) + '\n')

    # init
    val_log.write("total image count: " + str(len(final_pred)) + '\n')
    val_log.write("init pred cluster count: " + str(len(set(final_pred))) + '\n')
    val_log.write("init label class count: " + str(len(set(labels))) + '\n')

    # p0, r0, f0 = bcubed(np.array(labels), np.array(final_pred))
    # val_log.write("after remove precision: ", round(p0, 4))
    # val_log.write("after remove recall: ", round(r0, 4))
    # val_log.write("after remove f score: ", round(f0, 4))

    # construct dict
    final_pred_dict = defaultdict(list)
    for k, v in enumerate(final_pred):
        final_pred_dict[v].append(k)
    # print(len(final_pred_dict))
    val_log.write("init pred cluster count: " + str(len(final_pred_dict)) + '\n')

    # remove final_pred lower than cluster size
    final_pred_rm_dict = final_pred_dict
    old_keys = list(final_pred_rm_dict.keys())
    for i in old_keys:
        if len(final_pred_rm_dict[i]) <= cluster_size:
            del final_pred_rm_dict[i]
    val_log.write("after remove pred cluster count: " + str(len(final_pred_rm_dict)) + '\n')

    # after remove
    labels_after_remove = []
    pred_after_remove = []
    for i in final_pred_rm_dict.keys():
        for j in final_pred_rm_dict[i]:
            labels_after_remove.append(labels[j])
            pred_after_remove.append(final_pred[j])
    val_log.write("after remove image count: " + str(len(labels_after_remove)) + '\n')

    val_log.write("after remove pred cluster count: " + str(len(set(pred_after_remove))) + '\n')
    val_log.write("after remove label class count: " + str(len(set(labels_after_remove))) + '\n')

    # calculate after remove precision/recall/f score.
    p1, r1, f1 = bcubed(np.array(labels_after_remove), np.array(pred_after_remove))
    val_log.write("after remove precision: " + str(round(p1, 4)) + '\n')
    val_log.write("after remove recall: " + str(round(r1, 4)) + '\n')
    val_log.write("after remove f score: " + str(round(f1, 4)) + '\n')

    # remove labels lower than cluster size
    final_pred_remove_dict = defaultdict(list)
    for k, v in enumerate(pred_after_remove):
        final_pred_remove_dict[v].append(k)

    labels_remove_dict = defaultdict(list)
    for k, v in enumerate(labels_after_remove):
        labels_remove_dict[v].append(k)

    # 1. cluster precision/recall/split
    pre_list = []
    rec_list = []
    num_list = []
    mostly_cluster_list = []
    for k in final_pred_remove_dict.keys():
        pred_k_after_remove_len = len(final_pred_remove_dict[k])
        classes_k_dict = defaultdict(list)
        for i, j in enumerate(final_pred_remove_dict[k]):
            classes_k_dict[labels_after_remove[j]].append(j)
        max_len = 0
        cur_label_after_remove = None
        for i in classes_k_dict.keys():
            if len(classes_k_dict[i]) >= max_len:
                max_len = len(classes_k_dict[i])
                cur_label_after_remove = labels_after_remove[classes_k_dict[i][0]]
                mostly_cluster_list.append(cur_label_after_remove)
        cur_pre = round(max_len / pred_k_after_remove_len, 2)
        cur_rec = round(max_len / len(labels_remove_dict[cur_label_after_remove]), 4)
        pre_list.append(cur_pre)
        rec_list.append(cur_rec)
        num_list.append(pred_k_after_remove_len)

    mostly_cluster_id_count = len(set(mostly_cluster_list))
    cluster_count = len(set(final_pred_remove_dict.keys()))
    Split = round(cluster_count / mostly_cluster_id_count, 4)
    # print('mostly_cluster_id_count: ', mostly_cluster_id_count)
    # print('cluster_count: ', cluster_count)

    val_log.write("4.6-1, after remove cluster_count: " + str(cluster_count) + '\n')
    val_log.write("4.6-2, after remove mostly_cluster_id_count: " + str(mostly_cluster_id_count) + '\n')
    val_log.write("4.1, after remove cluster precision: " + str(round(np.array(pre_list).mean(), 4)) + '\n')
    val_log.write("4.2, after remove cluster recall: " + str(round(np.array(rec_list).mean(), 4)) + '\n')
    val_log.write("4.3, after remove cluster split: " + str(Split) + '\n')

    # 4. person recall

    labels_dict = defaultdict(list)
    for k, v in enumerate(labels):
        labels_dict[v].append(k)
    labels_over_dict = labels_dict
    old_keys = list(labels_over_dict.keys())
    for i in old_keys:
        if len(labels_over_dict[i]) <= cluster_size:
            del labels_over_dict[i]
    classes_count = len(set(labels_over_dict))
    RP = round(mostly_cluster_id_count / classes_count, 4)
    val_log.write("4.6-3, after remove anno class count: " + str(classes_count) + '\n')
    val_log.write("4.4 after remove person recall: " + str(RP) + '\n')
    # print(mostly_cluster_id_count)
    # print(classes_count)

    # 5. image recall
    Nc = len(labels_after_remove)
    Nid = len(labels)
    IR = Nc / Nid

    val_log.write("4.5, after remove image recall: " + str(round(IR, 4)) + '\n')
    val_log.write("4.5-1, after remove image count: " + str(Nc) + '\n')
    val_log.write("4.5-2, total image count: " + str(Nid) + '\n')
    # print(Nc)
    # print(Nid)
    val_log.write('\n\n')

val_log.close()

