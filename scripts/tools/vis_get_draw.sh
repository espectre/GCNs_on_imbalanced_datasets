cd ../../

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=.

s1_folder=/home/zhiyuan.zfy/data/hgy_4days/ref_clusters/cluster_labeled
s2_folder=/home/zhiyuan.zfy/data/hgy_4days/0625_v1_visualize_after_filt_unlabeled

rm -fr output && mkdir -p output

for ctype in small middle big
do
    while read line
    do
        s1=`echo $line | cut -d " " -f1`
        s2=`echo $line | cut -d " " -f2`
        echo python tools/get_draw.py ${s1_folder}/${s1}.txt ${s2_folder}/${s2}.txt output/${ctype}_${s1}.jpg
        python tools/get_draw.py ${s1_folder}/${s1}.txt ${s2_folder}/${s2}.txt output/${ctype}_${s1}.jpg
    done < ${ctype}.txt
done