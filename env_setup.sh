#!/bin/sh

ENVS=$(conda env list | awk '{print $1}')

if [[ $ENVS = *"$1"* ]]; then
    echo "[PT INFO] \"$1\" already exists. Pass the installation"
else 
    echo "[PT INFO] Creating $1..."
    conda create -n $1 python=3.7 -y
    conda activate "$1"
    echo "[PT INFO] Done !"

    echo "[PT INFO] Dependecies..."
    conda install pytorch=1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia -y
    conda install -c anaconda h5py pyyaml -y
    conda install -c conda-forge sharedarray tensorboardx -y
    echo "[PT INFO] Done !"

    echo "[PT INFO] Installing cuda operations..."
    cd lib/pointops
    python3 setup.py install
    cd ../..
    echo "[PT INFO] Done !"

    NVCC="$(nvcc --version)"
    TORCH="$(python -c "import torch; print(torch.__version__)")"

    echo "[PT INFO] Finished the installation!"
    echo "[PT INFO] ========== Configurations =========="
    echo "$NVCC"
    echo "[PT INFO] PyTorch version: $TORCH"
    echo "[PT INFO] ===================================="

fi;