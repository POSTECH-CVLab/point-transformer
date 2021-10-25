#ifndef _BALLQUERY_CUDA_KERNEL
#define _BALLQUERY_CUDA_KERNEL
#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

void ballquery_cuda(int b, int n, int m, float radius, int nsample, at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_tensor);

void ballquery_cuda_fast(int b, int n, int m, float radius, int nsample, at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void ballquery_cuda_launcher(int b, int n, int m, float radius, int nsample, const float *xyz, const float *new_xyz, int *idx);

void ballquery_cuda_launcher_fast(int b, int n, int m, float radius, int nsample, const float *new_xyz, const float *xyz, int *idx, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
