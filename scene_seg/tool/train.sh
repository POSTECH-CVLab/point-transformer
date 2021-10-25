#!/bin/sh
export PYTHONPATH=./

PYTHON=python
dataset=$1
exp_name=$2
exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
config=config/${dataset}/${dataset}_${exp_name}.yaml

mkdir -p ${model_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp tool/train.sh tool/train.py ${config} ${exp_dir}

$PYTHON tool/train.py --config=${config} 2>&1 | tee ${model_dir}/train-$now.log


if [ ${dataset} = 's3dis' ]
then
  $PYTHON tool/test_s3dis.py --config=${config} 2>&1 | tee ${model_dir}/test-$now.log
fi
