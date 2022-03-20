#!/bin/bash

source activate uai_36
source path.sh
#echo "Activated env"

data_dir=$1
test_split=$2
mode=$3

  python evaluate/evaluate_FDR.py \
    --mode ${mode} \
    --trials_root ${data_dir}/trials/${test_split} \
    --data_root ${data_dir}/data/${test_split} \
    --pred_root ${data_dir}/transformed_embeddings/${test_split}/${mode} \
    --scores_root ${data_dir}/scores/${test_split} \
    --eval_xvector
