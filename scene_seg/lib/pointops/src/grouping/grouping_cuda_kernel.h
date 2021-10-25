#ifndef _GROUPING_CUDA_KERNEL
#define _GROUPING_CUDA_KERNEL
#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

void grouping_forward_cuda(int b, int c, int n, int m, int nsample, at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out);
void grouping_backward_cuda(int b, int c, int n, int m, int nsample, at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor grad_points_tensor);

void grouping_forward_cuda_fast(int b, int c, int n, int npoints, int nsample, at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void grouping_forward_cuda_launcher(int b, int c, int n, int m, int nsample, const float *points, const int *idx, float *out);
void grouping_backward_cuda_launcher(int b, int c, int n, int m, int nsample, const float *grad_out, const int *idx, float *grad_points);

void grouping_forward_cuda_launcher_fast(int b, int c, int n, int npoints, int nsample, const float *points, const int *idx, float *out);

#ifdef __cplusplus
}
#endif
#endif
