cd ../../

export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=.

#python3 tools/test_knn.py \
#    --name /workspace/data/semi_celeb/features/train_norm_fea_50 \
#    --prefix /workspace/data/semi_celeb \
#    --knn_method faiss_gpu \
#    --no_normalize \
#    --knn 80

## annotation
#for feat in train train_512 train_1024
#do
#  echo now is $feat
#  for k in 10 20 50 100 200
#  do
#    echo now is ${k}
#    timer_start=`date "+%Y-%m-%d %H:%M:%S"`
#    timer_end=`date "+%Y-%m-%d %H:%M:%S"`
#    python3 tools/get_knn.py \
#        --data_prefix /workspace/data/public/celeb_ltc/features \
#        --feats_file ${feat}.npy \
#        --topk ${k}
#    duration=`echo eval $(($(date +%s -d "${timer_end}") - $(date +%s -d "${timer_start}"))) | awk '{t=split("60 s 60 m 24 h 999 d",a);for(n=1;n<t;n+=2){if($1==0)break;s=$1%a[n]a[n+1]s;$1=int($1/a[n])}print s}'`
#    echo "耗时： $duration"
#  done
#done

## annotation
#python3 tools/get_knn.py \
#    --data_prefix /workspace/data/public/deepfashion/feature \
#    --feats_file deepfashion_test.npy \
#    --topk 20

# random celeb
for head_class_num in 200 500
do
  for tail_class_size in 3
  do
    cur_dir=LEAP_random_deepfashion_${head_class_num}_${tail_class_size}
    echo $cur_dir
    python3 tools/get_knn.py \
        --data_prefix /workspace/data/20200811_clean/random_deepfashion/${cur_dir} \
        --feats_file test_feat_deepfashion_random.npy \
        --topk 20
  done
done