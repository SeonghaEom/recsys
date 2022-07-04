#bin/bash

if [[ $1 == NISER ]]; then
    CUDA_VISIBLE_DEVICES=$3 nohup python -u scripts/main_niser.py --dataset-dir ../datasets/$2 > niser_$2.log &
elif [[ $1 == SRGNN ]]; then
    nohup python -u scripts/main_srgnn.py --dataset-dir ../datasets/$2
elif [[ $1 == LESSR ]]; then
    python -u scripts/main_lessr.py --dataset-dir ../datasets/$2 --num-layers 1
elif [[ $1 == MSGIFSR ]]; then
    CUDA_VISIBLE_DEVICES=$3 python -u scripts/main_msgifsr.py --dataset_dir=../datasets/$2 --num-layers=$4 --num-workers=$5 --order=$6 --reducer=$7 $8 --embedding_pretrained=../datasets/FA_$9_feat.npy
fi