#!/bin/bash

source activate uai_36
source path.sh
#echo "Activated env"

test_split=$1
exp_id=$2
epoch=$3
mode=$4  #'UAI', 'NLDR' 'UAI-AT', 'UAI-MTL', 'AT', 'MTL'
model_dir=/proj/rperi/UAI/saved_models/exp_${exp_id}
pred_dir=/data/rperi/uai_pytorch/predictions_cv_$1/
mkdir -p ${pred_dir}

  # Peroform model inference 
  python predict.py \
	--config_path configs/exp_test.cfg \
	--exp_id ${exp_id} \
	--epoch ${epoch} \
    --mode ${mode} \
	--data_name  combined \
	--inp_feats /proj/rperi/UAI/data/data_CommonVoice_${test_split}//test_data.npy \
	--out_root ${pred_dir} \
	--checkpoint_file ${model_dir}/Epoch_${epoch}.pt
