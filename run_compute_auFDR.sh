#!/bin/bash

source activate uai_36
source path.sh
#echo "Activated env"

test_split=$1
exp_id=$2
epoch=$3

  python evaluate/evaluate_FDR.py \
    --test_split ${test_split} \
    --exp_id ${exp_id} \
    --epoch ${epoch} \
    --trials_root /proj/rperi/UAI/data/trials/CommonVoice/${test_split} \
    --data_root /proj/rperi/UAI/data/data_CommonVoice_${test_split} \
    --scores_root /data/rperi/uai_pytorch/scores_CommonVoice_${test_split} \
    --eval_xvector \
    --xvector_type balanced
          
