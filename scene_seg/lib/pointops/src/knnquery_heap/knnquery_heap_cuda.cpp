#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <ATen/cuda/CUDAContext.h>

#include "knnquery_heap_cuda_kernel.h"

extern THCState *state;

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)


void knnquery_heap_cuda(int b, int n, int m, int nsample, at::Tensor xyz_tensor, at::Tensor new_xyz_tensor, at::Tensor idx_tensor, at::Tensor dist2_tensor)
{
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(xyz_tensor);

    const float *new_xyz = new_xyz_tensor.data_ptr<float>();
    const float *xyz = xyz_tensor.data_ptr<float>();
    int *idx = idx_tensor.data_ptr<int>();
    float *dist2 = dist2_tensor.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    knnquery_heap_cuda_launcher(b, n, m, nsample, xyz, new_xyz, idx, dist2, stream);
}
