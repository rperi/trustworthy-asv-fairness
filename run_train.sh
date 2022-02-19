#!/bin/bash

#source activate uai_36
#echo "Activated env"
exp=$1

python train.py --config_path configs/exp_${1}.cfg --data_root /proj/rperi/UAI/data/data_FairVoice_train2/ --weights_root /proj/rperi/UAI/saved_models/ --logs_root /proj/rperi/UAI/logs/

