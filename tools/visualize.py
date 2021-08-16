#coding:utf-8 
import sys
import pickle
import numpy as np
from utils.misc import clusters_pickle2txts

if __name__ == "__main__":
    pkl_file = sys.argv[1]

    clusters_pickle2txts(pkl_file)

