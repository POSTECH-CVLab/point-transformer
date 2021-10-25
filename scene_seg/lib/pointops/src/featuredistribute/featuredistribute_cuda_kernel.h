#ifndef _FEATUREDISTRIBUTE_CUDA_KERNEL
#define _FEATUREDISTRIBUTE_CUDA_KERNEL
#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

void featuredistribute_cuda(int b, int n, int m, at::Tensor max_xyz_tensor, at::Tensor xyz_tensor, at::Tensor distribute_idx_tensor);
void featuregather_forward_cuda(int b, int n, int m, int c, at::Tensor max_feature_tensor, at::Tensor distribute_idx_tensor, at::Tensor distribute_feature_tensor);
void featuregather_backward_cuda(int b, int n, int m, int c, at::Tensor grad_distribute_feature_tensor, at::Tensor distribute_idx_tensor, at::Tensor grad_max_feature_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void featuredistribute_cuda_launcher(int b, int n, int m, const float *max_xyz, const float *xyz, int *distribute_idx, cudaStream_t stream);
void featuregather_forward_cuda_launcher(int b, int n, int m, int c, const float *max_feature, const int *distribute_idx, float *distribute_feature, cudaStream_t stream);
void featuregather_backward_cuda_launcher(int b, int n, int m, int c, const float *grad_distribute_feature, const int *distribute_idx, float *grad_max_feature, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
