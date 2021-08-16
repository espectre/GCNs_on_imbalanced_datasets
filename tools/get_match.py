#coding:utf-8
import sys
import os
import glob
from multiprocessing import Pool
import functools

def get_faceid_set(filename):
    faceid_set = set()
    for line in open(filename):
        line = line.strip()
        faceid_set.add(line)
    return faceid_set

def get_cmp(c1_dict, item):
    cid2, res2 = item
    for cid1, res1 in c1_dict.items():
        iou = len((res1 & res2)) / len((res1 | res2))
        if iou < 0.7:
            continue
        if len(res1) > 200 and len(res2) > 200:  # the big cluster restrict
            return ('big', cid1, cid2)
        elif 20 < len(res1) < 200 and 20 < len(res2) < 200:  # the middle cluster restrict
            return ('middle', cid1, cid2)
        elif len(res1) < 20 and len(res2) < 20:  # the small cluster restrict
            return ('small', cid1, cid2)
        else:
            pass
    return None
    # 只选一个不求全

if __name__ == "__main__":
    cluster1_path, cluster2_path = sys.argv[1], sys.argv[2]

    c1_dict, c2_dict = {}, {}
    for filename in glob.glob('%s/cluster*.txt'%cluster1_path):
        cid = os.path.splitext(os.path.basename(filename))[0]
        res = get_faceid_set(filename)
        c1_dict[cid] = res

    for filename in glob.glob('%s/cluster*.txt'%cluster2_path):
        cid = os.path.splitext(os.path.basename(filename))[0]
        res = get_faceid_set(filename)
        c2_dict[cid] = res

    pool = Pool(50)
    get_cmp_partial = functools.partial(get_cmp, c1_dict)
    res = pool.map(get_cmp_partial, list(c2_dict.items()))

    f1 = open('big.txt', 'w')
    f2 = open('middle.txt', 'w')
    f3 = open('small.txt', 'w')
    for each in res:
        if each == None:
            continue
        cluster_type, cid1, cid2 = each
        if cluster_type == 'big':
            print (cid1, cid2, file=f1)
        elif cluster_type == 'middle':
            print (cid1, cid2, file=f2)
        else:
            print (cid1, cid2, file=f3)
    f1.close()
    f2.close()
    f3.close()