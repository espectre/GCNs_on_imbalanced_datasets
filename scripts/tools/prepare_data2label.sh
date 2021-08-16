cd ../../

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=.

python tools/prepare_data_to_label.py \
    --body_csv /home/zhiyuan.zfy/data/data_to_label/hgy_damo_body_20200627.csv \
    --face_csv /home/zhiyuan.zfy/data/data_to_label/hgy_damo_face_20200627.csv \
    --pkl_path /home/zhiyuan.zfy/data/data_to_label/out_hardpos.pkl \
    --unique_list /home/zhiyuan.zfy/data/data_to_label/unique_face_ids.npy \
    --src_prefix /home/zhiyuan.zfy/data/data_to_label/person \
    --dst_prefix /home/zhiyuan.zfy/data/data_to_label/person_renamed \
    --src_face_dir /home/zhiyuan.zfy/data/data_to_label/pics \
    --src_body_dir /home/zhiyuan.zfy/data/data_to_label/person_renamed \
    --dst_face_dir /home/zhiyuan.zfy/data/data_to_label/used_pics/face \
    --dst_body_dir /home/zhiyuan.zfy/data/data_to_label/used_pics/body \
    --oss_prefix dabiao/zhiyuan/hgy/hardsamples \
    --output_path /home/zhiyuan.zfy/data/data_to_label/hardsamples/data \
    # --process_body_imgs
