#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <THC/THC.h>

#include "grouping_cuda_kernel.h"

extern THCState *state;

void grouping_forward_cuda(int b, int c, int n, int m, int nsample, at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor)
{
    const float *points = points_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();
    float *out = out_tensor.data_ptr<float>();
    grouping_forward_cuda_launcher(b, c, n, m, nsample, points, idx, out);
}

void grouping_backward_cuda(int b, int c, int n, int m, int nsample, at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor grad_points_tensor)
{
    float *grad_points = grad_points_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();
    const float *grad_out = grad_out_tensor.data_ptr<float>();
    grouping_backward_cuda_launcher(b, c, n, m, nsample, grad_out, idx, grad_points);
}

void grouping_forward_cuda_fast(int b, int c, int n, int npoints, int nsample, at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor) {

    const float *points = points_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();
    float *out = out_tensor.data_ptr<float>();
    grouping_forward_cuda_launcher_fast(b, c, n, npoints, nsample, points, idx, out);
}