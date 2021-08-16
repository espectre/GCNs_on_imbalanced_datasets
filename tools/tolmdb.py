#coding:utf-8 
import sys
import pickle
import numpy as np
import lmdb
if __name__ == "__main__":
    infile, outfile = sys.argv[1], sys.argv[2]

    arr = np.load(infile, allow_pickle=True)
    env = lmdb.open(outfile, map_size=1099511627776)
    txn = env.begin(write=True)

    for idx, data in enumerate(arr):
        data_str = pickle.dumps(data)
        txn.put(str(idx).encode('utf-8'), data_str)
        # if idx % 1000 == 0:
        #     print("==>", idx)
    txn.commit()
    env.close()

