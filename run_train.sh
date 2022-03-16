#!/bin/bash

#source activate uai_36
source path.sh

data_dir=$1 #<data_dir>
mode=$2 #'UAI', 'NLDR' 'UAI-AT', 'UAI-MTL', 'AT', 'MTL'

python train.py \
        --config_path configs/exp_${mode}.cfg \
        --data_root ${data_dir}/embed-train_val/ \
        --weights_root ${data_dir}/saved_models/ \
        --logs_root ${data_dir}/logs/

