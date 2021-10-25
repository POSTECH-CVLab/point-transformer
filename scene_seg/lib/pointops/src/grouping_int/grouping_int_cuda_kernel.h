#ifndef _GROUPING_INT_CUDA_KERNEL
#define _GROUPING_INT_CUDA_KERNEL
#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

void grouping_int_forward_cuda(int b, int c, int n, int m, int nsample, at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out);

void grouping_int_forward_cuda_fast(int b, int c, int n, int m, int nsample, at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void grouping_int_forward_cuda_launcher(int b, int c, int n, int m, int nsample, const long int *points, const int *idx, long int *out);

void grouping_int_forward_cuda_launcher_fast(int b, int c, int n, int npoints, int nsample, const long int *points, const int *idx, long int *out);

#ifdef __cplusplus
}
#endif
#endif
