cd ../../

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=.

python tools/tolmdb.py \
/workspace/data/20200811_clean/random_deepfashion/LEAP_random_deepfashion_500_3/train_feat_deepfashion_random.npy \
/workspace/data/20200811_clean/random_deepfashion/LEAP_random_deepfashion_500_3/train_feat_deepfashion_random.lmdb
