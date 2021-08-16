cd ../../

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=.

python tools/generate_semi_data.py \
/workspace/data/semi_celeb/labels/label_node_list_200.npy \
/workspace/data/semi_celeb/labels/pseudo_labels_400.npy \
/workspace/data/semi_celeb/labels/train_label_400.npy \
/workspace/data/semi_celeb/features/train_norm_fea_400/faiss_gpu_k_80.npy \
/workspace/data/semi_celeb/labels/semi_label_200_200.npy