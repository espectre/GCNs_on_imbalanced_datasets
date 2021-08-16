export CUDA_VISIBLE_DEVICES=0,1,2,3
model_tag=params/ex1_baseline_sup_random_celeb/baseline_sup_random_celeb

echo now is train stage
for head_class_num in 200 500 1000 2000
do
  for tail_class_size in 3 5
  do
    cur_dir=LEAP_random_celeb_${head_class_num}_${tail_class_size}
    echo $cur_dir
    cur_save_dir=${model_tag}/${cur_dir}
    echo ${cur_save_dir}
    rm -fr ${cur_save_dir} && mkdir -p ${cur_save_dir}
    python -m torch.distributed.launch --nproc_per_node=1 --master_port 18115 train_dist.py \
           --feat_path /workspace/data/20200811_clean/random_celeb/${cur_dir}/train_feat_celeb_random.lmdb \
           --knn_graph_path /workspace/data/20200811_clean/random_celeb/${cur_dir}/train_feat_celeb_random/knn_index_200.npy \
           --density_path /workspace/data/20200811_clean/random_celeb/${cur_dir}/train_density.npy \
           --label_path /workspace/data/20200811_clean/random_celeb/${cur_dir}/train_label_celeb_random.npy \
           --logs-dir ${cur_save_dir} --print_freq 20 --workers 2 \
           --rand_val 0 --rand_twice 0 --resample 0 --is_density 0 --clique_size 0 --batch_size 16 \
           --lr 0.01 --epochs 4 --features 256 --k-at-hop 80 10  --active_connection 10 \
           --model_drop gcn --block_size 3 --drop_prob 0.7 \
           --loss normal --ratio 0.5 --gamma 2 --alpha 0.75 \
           --mode sup | tee ${cur_save_dir}/log.out

    echo now is test stage
    for idx in $(seq 4 4)
    do
      echo $idx
      modelfile=${cur_save_dir}/epoch_${idx}.ckpt
      echo now is $modelfile
      python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 18125 test_dist.py \
             --val_feat_path /workspace/data/public/celeb_ltc/features/test.lmdb \
             --val_knn_graph_path /workspace/data/public/celeb_ltc/features/test/knn_index_80.npy \
             --val_label_path /workspace/data/public/celeb_ltc/labels/label_test.npy \
             --checkpoint $modelfile --batch_size 16 --max_size 300 \
             --features 256 --workers 2 --print_freq 20  --k-at-hop 79 5 \
             --active_connection 5 | tee -a $cur_save_dir/eval.log
    done
  done
done