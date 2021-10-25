import os
import torch
from torch.utils.cpp_extension import load

cwd = os.path.dirname(os.path.realpath(__file__))
gpu_path = os.path.join(cwd, 'gpu')

if torch.cuda.is_available():
    gpu = load('gpconv_cuda', [
        os.path.join(gpu_path, 'operator.cpp'),
        os.path.join(gpu_path, 'assign_score_withk_gpu.cu'),
    ], build_directory=gpu_path, verbose=False)


