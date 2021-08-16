#coding:utf-8 
import sys
import time
import numpy as np
from multiprocessing import Pool 
import functools


def update_node_with_labels(label_node_list, pseudo_labels, labels, knns):
    top_k = knns.shape[1]
    label_node_list_ = set(label_node_list)
    start = time.time()
    for i in range(len(label_node_list)):
        index = label_node_list[i]

        for j in range(top_k):
            if knns[index,j] in label_node_list_:
                if labels[index] == labels[knns[index,j]]:
                    pseudo_labels[index,j] = 1
                else:
                    pseudo_labels[index,j] = 0

        if i % 200000 == 0:
            print("Updated labels on Row: ", index)
            print(time.time() - start)
    return pseudo_labels


def load_files(label_node_list_file, pseudo_labels_file, labels_file, knns_file):
    label_node_list = np.load(label_node_list_file).reshape(-1)
    pseudo_labels = np.load(pseudo_labels_file)
    labels = np.load(labels_file).reshape(-1)
    knns = np.load(knns_file)
    return label_node_list, pseudo_labels, labels, knns

if __name__ == "__main__":
    label_node_list_file, pseudo_labels_file, labels_file, knns_file, outfile = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    label_node_list, pseudo_labels, labels, knns = load_files(label_node_list_file, pseudo_labels_file, labels_file, knns_file)
   
    new_pseudo_labels = update_node_with_labels(label_node_list, pseudo_labels, labels, knns)

    np.save(outfile, new_pseudo_labels)
    print("Pseudo Labels Updated!!!")

