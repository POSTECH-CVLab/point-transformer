#include "../cuda_utils.h"
#include "featuredistribute_cuda_kernel.h"

__global__ void featuredistribute_cuda_kernel(int b, int n, int m, const float *max_xyz, const float *xyz, int *distribute_idx) {
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m) return;

    max_xyz += bs_idx * n * 3;
    xyz += bs_idx * m * 3 + pt_idx * 3;
    distribute_idx += bs_idx * m + pt_idx;

    float x = xyz[0];
    float y = xyz[1];
    float z = xyz[2];

    float min_dist2 = 100000;
    int min_dist_idx = -1;
    for (int k = 0; k < n; ++k) {
        float max_x = max_xyz[k * 3 + 0];
        float max_y = max_xyz[k * 3 + 1];
        float max_z = max_xyz[k * 3 + 2];
        float d2 = (max_x - x) * (max_x - x) + (max_y - y) * (max_y - y) + (max_z - z) * (max_z - z);
        if (d2 < min_dist2){
            min_dist_idx = k;
            min_dist2 = d2;
        }
    }
    distribute_idx[0] = min_dist_idx;
}


void featuredistribute_cuda_launcher(int b, int n, int m, const float *max_xyz, const float *xyz, int *distribute_idx, cudaStream_t stream) {
    // param max_xyz: (b, n, 3)
    // param xyz: (b, m, 3)
    // return distribute_idx: (b, m)

    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    featuredistribute_cuda_kernel<<<blocks, threads, 0, stream>>>(b, n, m, max_xyz, xyz, distribute_idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void featuregather_forward_cuda_kernel(int b, int n, int m, int c, const float *max_feature, const int *distribute_idx, float *distribute_feature) {
    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || c_idx >= c || pt_idx >= m) return;

    max_feature += bs_idx * c * n + c_idx * n;
    distribute_idx += bs_idx * m + pt_idx;
    distribute_feature += bs_idx * c * m + c_idx * m + pt_idx;

    int idx = distribute_idx[0];
    distribute_feature[0] = max_feature[idx];
}


void featuregather_forward_cuda_launcher(int b, int n, int m, int c, const float *max_feature, const int *distribute_idx, float *distribute_feature, cudaStream_t stream){
    // param max_feature: (b, c, n)
    // param distribute_idx: (b, m)
    // return distribute_feature: (b, c, m)

    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    featuregather_forward_cuda_kernel<<<blocks, threads, 0, stream>>>(b, n, m, c, max_feature, distribute_idx, distribute_feature);
    // cudaDeviceSynchronize();  // for using printf in kernel function

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


__global__ void featuregather_backward_cuda_kernel(int b, int n, int m, int c, const float *grad_distribute_feature, const int *distribute_idx, float *grad_max_feature){
    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(bs_idx >= b || c_idx >= c || pt_idx >= m) return;

    grad_distribute_feature += bs_idx * c * m + c_idx * m + pt_idx;
    distribute_idx += bs_idx * m + pt_idx;
    grad_max_feature += bs_idx * c * n + c_idx * n;

    int idx = distribute_idx[0];
    atomicAdd(grad_max_feature + idx, grad_distribute_feature[0]);
}


void featuregather_backward_cuda_launcher(int b, int n, int m, int c, const float *grad_distribute_feature, const int *distribute_idx, float *grad_max_feature, cudaStream_t stream){
    // param grad_distribute_feature: (b, c, m)
    // param distribute_idx: (b, m)
    // return grad_max_feature: (b, c, n)

    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    featuregather_backward_cuda_kernel<<<blocks, threads, 0, stream>>>(b, n, m, c, grad_distribute_feature, distribute_idx, grad_max_feature);
    // cudaDeviceSynchronize();  // for using printf in kernel function

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}