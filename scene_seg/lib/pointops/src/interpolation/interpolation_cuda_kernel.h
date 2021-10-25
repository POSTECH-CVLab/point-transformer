#ifndef _INTERPOLATION_CUDA_KERNEL
#define _INTERPOLATION_CUDA_KERNEL
#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

void nearestneighbor_cuda(int b, int n, int m, at::Tensor unknown_tensor, at::Tensor known_tensor, at::Tensor dist2_tensor, at::Tensor idx_tensor);
void interpolation_forward_cuda(int b, int c, int m, int n, at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor weight_tensor, at::Tensor out_tensor);
void interpolation_backward_cuda(int b, int c, int n, int m, at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor weight_tensor, at::Tensor grad_points_tensor);

void nearestneighbor_cuda_fast(int b, int n, int m, at::Tensor unknown_tensor, at::Tensor known_tensor, at::Tensor dist2_tensor, at::Tensor idx_tensor);
void interpolation_forward_cuda_fast(int b, int c, int m, int n, at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor weight_tensor, at::Tensor out_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void nearestneighbor_cuda_launcher(int b, int n, int m, const float *unknown, const float *known, float *dist2, int *idx);
void interpolation_forward_cuda_launcher(int b, int c, int m, int n, const float *points, const int *idx, const float *weight, float *out);
void interpolation_backward_cuda_launcher(int b, int c, int n, int m, const float *grad_out, const int *idx, const float *weight, float *grad_points);

void nearestneighbor_cuda_launcher_fast(int b, int n, int m, const float *unknown, const float *known, float *dist2, int *idx);
void interpolation_forward_cuda_launcher_fast(int b, int c, int m, int n, const float *points, const int *idx, const float *weight, float *out);

#ifdef __cplusplus
}
#endif
#endif
