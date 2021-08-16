#coding:utf-8 
import sys
import numpy as np
from multiprocessing import Pool 
import functools

thres = 0.55

def get_discrete(thres, data_info):
    node_id, dist = data_info
    node_id = node_id.astype(np.int32)

    dist[np.where(dist<thres)[0]] = -1  # 本应是label1，这里暂填-1, 它不会参与到下面的 >thres_high 中
    dist[np.where(dist>=thres)[0]] = 0
    dist[np.where(dist==-1)[0]] = 1

    return dist.astype(np.int32)

if __name__ == "__main__":
    knnfile, outfile, thres = sys.argv[1], sys.argv[2], float(sys.argv[3])
    if knnfile.endswith('npz'):
        knn_arr = np.load(knnfile)['data']
    else:
        knn_arr = np.load(knnfile)

    pool = Pool(54)
    print("now is thres", thres)
    get_discrete_partial = functools.partial(get_discrete, thres)
    res_list = pool.map(get_discrete_partial, knn_arr)
    pool.close()
    pool.join()

    res_arr = np.array(res_list)
    #np.savez(outfile, data=res_arr)
    np.save(outfile+'_thres_'+str(thres), res_arr)

