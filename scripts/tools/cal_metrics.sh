cd ../../
export PYTHONPATH=.

python tools/cal_metrics.py \
    --work_dir /workspace/code/ali-gcn-clustering/ali_lgcn/params/u_5_0.2_sup_hp_9w/ \
    --val_log_file val_log_deploy.txt \
    --pred_file_path final_pred.npy \
    --label_file_path labels.npy \
    --cluster_size_list "1, 2, 4"


