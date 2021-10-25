#ifndef _LABELSTAT_CUDA_KERNEL
#define _LABELSTAT_CUDA_KERNEL
#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

void labelstat_and_ballquery_cuda_fast(int b, int n, int m, float radius, int nsample, int nclass,
    at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor label_stat_tensor, at::Tensor idx_tensor, at::Tensor new_label_stat_tensor);

void labelstat_ballrange_cuda_fast(int b, int n, int m, float radius, int nclass,
    at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor label_stat_tensor, at::Tensor new_label_stat_tensor);

void labelstat_idx_cuda_fast(int b, int n, int m, int nsample, int nclass,
    at::Tensor label_stat_tensor, at::Tensor idx_tensor, at::Tensor new_label_stat_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void labelstat_and_ballquery_cuda_launcher_fast(int b, int n, int m, float radius, int nsample, int nclass, \
    const float *new_xyz, const float *xyz, const int *label_stat, int *idx, int *new_label_stat, cudaStream_t stream);

void labelstat_ballrange_cuda_launcher_fast(int b, int n, int m, float radius, int nclass, \
    const float *new_xyz, const float *xyz, const int *label_stat, int *new_label_stat, cudaStream_t stream);

void labelstat_idx_cuda_launcher_fast(int b, int n, int m, int nsample, int nclass, \
    const int *label_stat, const int *idx, int *new_label_stat, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
