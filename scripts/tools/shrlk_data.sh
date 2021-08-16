cd ../../

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=.

python tools/shrlk_data.py \
    --name testset_gn_ft_20200623 \
    --prefix /home/zhiyuan.zfy/data/renlianku_sh/origin \
    --dim 512 \
    --doc_amount 1 \
    --no_normalize \
    --sort
