cd ../../

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=.

python tools/extract_labels_from_jasons.py \
    --pkl_path /home/zhiyuan.zfy/data/data_to_label/out_hardpos.pkl \
    --label_path /home/zhiyuan.zfy/data/data_to_label/part_labels_20200722/50cfd8cd-14ce-454a-be00-27422ccd8b0e/hardsamples/data \
    --output_path /home/zhiyuan.zfy/data/data_to_label/labels
