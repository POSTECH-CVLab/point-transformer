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

if [ ${dataset} = 's3dis' ]
then
  cp tool/test.sh tool/test_s3dis.py ${config} ${exp_dir}
  $PYTHON tool/test_s3dis.py --config=${config} 2>&1 | tee ${model_dir}/test-$now.log
fi
