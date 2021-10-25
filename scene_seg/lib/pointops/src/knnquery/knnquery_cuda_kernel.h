#ifndef _KNNQUERY_CUDA_KERNEL
#define _KNNQUERY_CUDA_KERNEL

#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

void knnquery_cuda(int b, int n, int m, int nsample, at::Tensor xyz_tensor, at::Tensor new_xyz_tensor, at::Tensor idx_tensor, at::Tensor dist2_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void knnquery_cuda_launcher(int b, int n, int m, int nsample, const float *xyz, const float *new_xyz, int *idx, float *dist2, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif