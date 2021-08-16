export CUDA_VISIBLE_DEVICES=4,5,6,7
model_tag=params/ex1_baseline_sup_random_deepfashion/baseline_sup_random_deepfashion

echo now is train stage
for head_class_num in 200 500
do
  for tail_class_size in 3
  do
    cur_dir=LEAP_random_deepfashion_${head_class_num}_${tail_class_size}
    echo $cur_dir
    cur_save_dir=${model_tag}/${cur_dir}
    echo ${cur_save_dir}
    rm -fr ${cur_save_dir} && mkdir -p ${cur_save_dir}
    python -m torch.distributed.launch --nproc_per_node=1 --master_port 21001 train_dist.py \
           --feat_path /workspace/data/20200811_clean/random_deepfashion/${cur_dir}/train_feat_deepfashion_random.lmdb \
           --knn_graph_path /workspace/data/20200811_clean/random_deepfashion/${cur_dir}/train_feat_deepfashion_random/knn_index_20.npy \
           --density_path /workspace/data/20200811_clean/random_deepfashion/${cur_dir}/train_density.npy \
           --label_path /workspace/data/20200811_clean/random_deepfashion/${cur_dir}/train_label_deepfashion_random.npy \
           --logs-dir ${cur_save_dir} --print_freq 20 --workers 2 \
           --rand_val 0 --rand_twice 0 --is_density 0 --clique_size 0 --batch_size 16 \
           --lr 0.01 --epochs 4 --features 256 --k-at-hop 5 5  --active_connection 5 \
           --model_drop gcn --block_size 3 --drop_prob 0.7 \
           --loss normal --ratio 0.5 --gamma 2 --alpha 0.75 \
           --mode sup | tee ${cur_save_dir}/log.out

    echo now is test stage
    for idx in $(seq 4 4)
    do
      echo $idx
      modelfile=${cur_save_dir}/epoch_${idx}.ckpt
      echo now is $modelfile
      python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 21111 test_dist.py \
             --val_feat_path /workspace/data/public/deepfashion/feature/deepfashion_test.lmdb \
             --val_knn_graph_path /workspace/data/public/deepfashion/feature/deepfashion_test/knn_index_20.npy \
             --val_label_path /workspace/data/public/deepfashion/label/deepfashion_test.npy \
             --checkpoint $modelfile --batch_size 16 --max_size 50 --prop_step 0.5 \
             --features 256 --workers 2 --print_freq 20  --k-at-hop 5 5 \
             --active_connection 5 | tee -a $cur_save_dir/eval.log
    done

  done
done