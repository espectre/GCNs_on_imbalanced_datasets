#!/bin/bash
cd ../../

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=.

python tools/get_match.py /home/zhiyuan.zfy/data/hgy_4days/ref_clusters/cluster_labeled /home/zhiyuan.zfy/data/hgy_4days/0625_v1_visualize_after_filt_unlabeled