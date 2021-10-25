#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <ATen/cuda/CUDAContext.h>

#include "labelstat_cuda_kernel.h"

extern THCState *state;


#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

void labelstat_idx_cuda_fast(int b, int n, int m, int nsample, int nclass,
    at::Tensor label_stat_tensor, at::Tensor idx_tensor, at::Tensor new_label_stat_tensor)
{
    CHECK_INPUT(label_stat_tensor);
    CHECK_INPUT(idx_tensor);

    const int *label_stat = label_stat_tensor.data_ptr<int>();
    const int *idx = idx_tensor.data_ptr<int>();
    int *new_label_stat = new_label_stat_tensor.data_ptr<int>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    labelstat_idx_cuda_launcher_fast(b, n, m, nsample, nclass, label_stat, idx, new_label_stat, stream);
}

void labelstat_ballrange_cuda_fast(int b, int n, int m, float radius, int nclass,
    at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor label_stat_tensor, at::Tensor new_label_stat_tensor)
{
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(xyz_tensor);
    CHECK_INPUT(label_stat_tensor);

    const float *new_xyz = new_xyz_tensor.data_ptr<float>();
    const float *xyz = xyz_tensor.data_ptr<float>();
    const int *label_stat = label_stat_tensor.data_ptr<int>();
    int *new_label_stat = new_label_stat_tensor.data_ptr<int>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    labelstat_ballrange_cuda_launcher_fast(b, n, m, radius, nclass, new_xyz, xyz, label_stat, new_label_stat, stream);
}

void labelstat_and_ballquery_cuda_fast(int b, int n, int m, float radius, int nsample, int nclass,
    at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor label_stat_tensor, at::Tensor idx_tensor, at::Tensor new_label_stat_tensor)
{
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(xyz_tensor);
    CHECK_INPUT(label_stat_tensor);
    CHECK_INPUT(idx_tensor);

    const float *new_xyz = new_xyz_tensor.data_ptr<float>();
    const float *xyz = xyz_tensor.data_ptr<float>();
    const int *label_stat = label_stat_tensor.data_ptr<int>();
    int *idx = idx_tensor.data_ptr<int>();
    int *new_label_stat = new_label_stat_tensor.data_ptr<int>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    labelstat_and_ballquery_cuda_launcher_fast(b, n, m, radius, nsample, nclass, new_xyz, xyz, label_stat, idx, new_label_stat, stream);
}
