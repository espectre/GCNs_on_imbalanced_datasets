cd ../../

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=.

python tools/extract_info_from_csv.py \
    --body_csv /home/zhiyuan.zfy/data/data_to_label/hgy_damo_body_20200627.csv \
    --face_csv /home/zhiyuan.zfy/data/data_to_label/hgy_damo_face_20200627.csv \
    --unique_list /home/zhiyuan.zfy/data/data_to_label/unique_face_ids.npy \
    --output_path /home/zhiyuan.zfy/data/data_to_label/selected_samples
