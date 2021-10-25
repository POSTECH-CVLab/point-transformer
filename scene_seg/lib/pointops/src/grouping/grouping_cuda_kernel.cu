#include "../cuda_utils.h"
#include "grouping_cuda_kernel.h"

// input: points(b, c, n) idx(b, m, nsample)
// output: out(b, c, m, nsample)
__global__ void grouping_forward_cuda_kernel(int b, int c, int n, int m, int nsample, const float *points, const int *idx, float *out)
{
    int batch_index = blockIdx.x;
    points += batch_index * n * c;
    idx += batch_index * m * nsample;
    out += batch_index * m * nsample * c;
    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    const int stride = blockDim.y * blockDim.x;
    for (int i = index; i < c * m; i += stride)
    {
        const int l = i / m;
        const int j = i % m;
        for (int k = 0; k < nsample; ++k)
        {
            int ii = idx[j * nsample + k];
            out[(l * m + j) * nsample + k] = points[l * n + ii];
        }
    }
}

// input: grad_out(b, c, m, nsample), idx(b, m, nsample)
// output: grad_points(b, c, n)
__global__ void grouping_backward_cuda_kernel(int b, int c, int n, int m, int nsample, const float *grad_out, const int *idx, float *grad_points)
{
    int batch_index = blockIdx.x;
    grad_out += batch_index * m * nsample * c;
    idx += batch_index * m * nsample;
    grad_points += batch_index * n * c;
    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    const int stride = blockDim.y * blockDim.x;
    for (int i = index; i < c * m; i += stride)
    {
        const int l = i / m;
        const int j = i % m;
        for (int k = 0; k < nsample; ++k)
        {
            int ii = idx[j * nsample + k];
            atomicAdd(grad_points + l * n + ii, grad_out[(l * m + j) * nsample + k]);
        }
    }
}

void grouping_forward_cuda_launcher(int b, int c, int n, int m, int nsample, const float *points, const int *idx, float *out)
{
    grouping_forward_cuda_kernel<<<b, opt_block_config(m, c), 0>>>(b, c, n, m, nsample, points, idx, out);
}

void grouping_backward_cuda_launcher(int b, int c, int n, int m, int nsample, const float *grad_out, const int *idx, float *grad_points)
{
    grouping_backward_cuda_kernel<<<b, opt_block_config(m, c), 0>>>(b, c, n, m, nsample, grad_out, idx, grad_points);
}

// input: points(b, c, n) idx(b, npoints, nsample)
// output: out(b, c, npoints, nsample)
__global__ void grouping_forward_cuda_kernel_fast(int b, int c, int n, int npoints, int nsample, const float *__restrict__ points, const int *__restrict__ idx, float *__restrict__ out) {
    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int pt_idx = index / nsample;
    if (bs_idx >= b || c_idx >= c || pt_idx >= npoints) return;

    int sample_idx = index % nsample;

    idx += bs_idx * npoints * nsample + pt_idx * nsample + sample_idx;
    int in_idx = bs_idx * c * n + c_idx * n + idx[0];
    int out_idx = bs_idx * c * npoints * nsample + c_idx * npoints * nsample + pt_idx * nsample + sample_idx;

    out[out_idx] = points[in_idx];
}

// input: points(b, c, n) idx(b, npoints, nsample)
// output: out(b, c, npoints, nsample)
void grouping_forward_cuda_launcher_fast(int b, int c, int n, int npoints, int nsample, const float *points, const int *idx, float *out) {

    cudaError_t err;

    dim3 blocks(DIVUP(npoints * nsample, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    grouping_forward_cuda_kernel_fast<<<blocks, threads, 0>>>(b, c, n, npoints, nsample, points, idx, out);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


