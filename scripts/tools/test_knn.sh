cd ../../

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH=.

#python3 tools/test_knn.py \
#    --name /workspace/data/semi_celeb/features/train_norm_fea_50 \
#    --prefix /workspace/data/semi_celeb \
#    --knn_method faiss_gpu \
#    --no_normalize \
#    --knn 80

python3 tools/test_knn.py \
    --name /workspace/data/hgy_17w/hgyA_feat_16to32 \
    --prefix /workspace/data/hgy_17w \
    --knn_method faiss_gpu \
    --no_normalize \
    --knn 80