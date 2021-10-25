#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <ATen/cuda/CUDAContext.h>

#include "featuredistribute_cuda_kernel.h"

extern THCState *state;

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)


void featuredistribute_cuda(int b, int n, int m, at::Tensor max_xyz_tensor, at::Tensor xyz_tensor, at::Tensor distribute_idx_tensor)
{
    CHECK_INPUT(max_xyz_tensor);
    CHECK_INPUT(xyz_tensor);

    const float *max_xyz = max_xyz_tensor.data_ptr<float>();
    const float *xyz = xyz_tensor.data_ptr<float>();
    int *distribute_idx = distribute_idx_tensor.data_ptr<int>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    featuredistribute_cuda_launcher(b, n, m, max_xyz, xyz, distribute_idx, stream);
}


void featuregather_forward_cuda(int b, int n, int m, int c, at::Tensor max_feature_tensor, at::Tensor distribute_idx_tensor, at::Tensor distribute_feature_tensor)
{
    CHECK_INPUT(max_feature_tensor);
    CHECK_INPUT(distribute_idx_tensor);

    const float *max_feature = max_feature_tensor.data_ptr<float>();
    const int *distribute_idx = distribute_idx_tensor.data_ptr<int>();
    float *distribute_feature = distribute_feature_tensor.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    featuregather_forward_cuda_launcher(b, n, m, c, max_feature, distribute_idx, distribute_feature, stream);
}


void featuregather_backward_cuda(int b, int n, int m, int c, at::Tensor grad_distribute_feature_tensor, at::Tensor distribute_idx_tensor, at::Tensor grad_max_feature_tensor)
{
    CHECK_INPUT(grad_distribute_feature_tensor);
    CHECK_INPUT(distribute_idx_tensor);

    const float *grad_distribute_feature = grad_distribute_feature_tensor.data_ptr<float>();
    const int *distribute_idx = distribute_idx_tensor.data_ptr<int>();
    float *grad_max_feature = grad_max_feature_tensor.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    featuregather_backward_cuda_launcher(b, n, m, c, grad_distribute_feature, distribute_idx, grad_max_feature, stream);
}