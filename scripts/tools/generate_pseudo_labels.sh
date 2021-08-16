cd ../../

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=.

python tools/generate_pseudo_labels.py \
/workspace/data/babybus/baby_bus_20201220_random_test/random_last_ids_feat_norm/knn_data_80.npz \
/workspace/data/babybus/baby_bus_20201220_random_test/pseudo_labels \
0.4