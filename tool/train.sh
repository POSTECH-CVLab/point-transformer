#!/bin/sh

export PYTHONPATH=./
eval "$(conda shell.bash hook)"
PYTHON=python

TRAIN_CODE=train.py
TEST_CODE=test.py

dataset=$1
exp_name=$2
exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml

mkdir -p ${model_dir} ${result_dir}
mkdir -p ${result_dir}/last
mkdir -p ${result_dir}/best
cp tool/train.sh tool/${TRAIN_CODE} ${config} tool/test.sh tool/${TEST_CODE} ${exp_dir}


now=$(date +"%Y%m%d_%H%M%S")
$PYTHON ${exp_dir}/${TRAIN_CODE} \
  --config=${config} \
  save_path ${exp_dir} \
  2>&1 | tee ${exp_dir}/train-$now.log
