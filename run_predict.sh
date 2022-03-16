#!/bin/bash

#source activate uai_36
source path.sh
#echo "Activated env"

data_dir=$1 # <data_dir>
test_split=$2 #'dev', 'test'
mode=$3  #'UAI', 'NLDR' 'UAI-AT', 'UAI-MTL', 'AT', 'MTL'

model_dir=${data_dir}/saved_models/
pred_dir=${data_dir}/transformed_embeddings/eval-${test_split}/${mode}
mkdir -p ${pred_dir}

  # Peroform model inference 
  python predict.py \
	--config_path configs/exp_test.cfg \
    --mode ${mode} \
	--inp_feats ${data_dir}/data/eval-${test_split}/test_data.npy \
	--out_dir ${pred_dir} \
	--checkpoint_file ${model_dir}/${mode}.pt
