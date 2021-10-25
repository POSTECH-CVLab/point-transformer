#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include "sampling_cuda_kernel.h"

extern THCState *state;

void gathering_forward_cuda(int b, int c, int n, int m, at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor)
{
    const float *points = points_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();
    float *out = out_tensor.data_ptr<float>();
    gathering_forward_cuda_launcher(b, c, n, m, points, idx, out);
}

void gathering_backward_cuda(int b, int c, int n, int m, at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor grad_points_tensor)
{

    const float *grad_out = grad_out_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();
    float *grad_points = grad_points_tensor.data_ptr<float>();
    gathering_backward_cuda_launcher(b, c, n, m, grad_out, idx, grad_points);
}

void furthestsampling_cuda(int b, int n, int m, at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor)
{
    const float *points = points_tensor.data_ptr<float>();
    float *temp = temp_tensor.data_ptr<float>();
    int *idx = idx_tensor.data_ptr<int>();
    furthestsampling_cuda_launcher(b, n, m, points, temp, idx);
}
