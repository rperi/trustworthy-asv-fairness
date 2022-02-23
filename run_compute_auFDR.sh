#!/bin/bash

source activate uai_36
source path.sh
#echo "Activated env"

data_dir=$1
test_split=$2
mode=$3

  python evaluate/evaluate_FDR.py \
    --test_split ${test_split} \
    --mode ${mode} \
    --trials_root ${data_dir}/trials/${test_split} \
    --data_root ${data_dir}/data/eval-${test_split} \
    --pred_root ${data_dir}/transformed_embeddings/eval-${test_split}/${mode} \
    --scores_root ${data_dir}/scores/eval-${test_split} \
    --eval_xvector
