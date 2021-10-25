#include "../cuda_utils.h"
#include "labelstat_cuda_kernel.h"

// input: new_xyz(b, m, 3) xyz(b, n, 3) label_stat(b, n, nclass)
// output: idx(b, m, nsample)  new_label_stat(b, m, nclass)
__global__ void labelstat_and_ballquery_cuda_kernel_fast(int b, int n, int m, float radius, int nsample, int nclass,
    const float *new_xyz, const float *xyz, const int *label_stat, int *idx, int *new_label_stat) {
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m) return;

    new_xyz += bs_idx * m * 3 + pt_idx * 3;
    xyz += bs_idx * n * 3;
    idx += bs_idx * m * nsample + pt_idx * nsample;
    label_stat += bs_idx * n * nclass;
    new_label_stat += bs_idx * m * nclass + pt_idx * nclass;

    for(int i = 0; i < nclass; i++){
        new_label_stat[i] = 0;
    }

    float radius2 = radius * radius;
    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];

    int cnt = 0;
    for (int k = 0; k < n; ++k) {
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        if (d2 < radius2){
            for(int i = 0; i < nclass; i++){
                new_label_stat[i] += label_stat[k * nclass + i];
            }
            if (cnt == 0){
                for (int l = 0; l < nsample; ++l) {
                    idx[l] = k;
                }
            }
            idx[cnt] = k;
            ++cnt;
            if (cnt >= nsample){
                break;
            }
        }
    }
}

void labelstat_and_ballquery_cuda_launcher_fast(int b, int n, int m, float radius, int nsample, int nclass,
    const float *new_xyz, const float *xyz, const int *label_stat, int *idx, int *new_label_stat, cudaStream_t stream) {
    // param new_xyz: (B, m, 3)
    // param xyz: (B, n, 3)
    // param idx: (B, m, nsample)

    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    labelstat_and_ballquery_cuda_kernel_fast<<<blocks, threads, 0, stream>>>(b, n, m, radius, nsample, nclass, new_xyz, xyz, label_stat, idx, new_label_stat);
    // cudaDeviceSynchronize();  // for using printf in kernel function

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

// input: new_xyz(b, m, 3) xyz(b, n, 3) label_stat(b, n, nclass)
// output: new_label_stat(b, m, nclass)
__global__ void labelstat_ballrange_cuda_kernel_fast(int b, int n, int m, float radius, int nclass,
    const float *new_xyz, const float *xyz, const int *label_stat, int *new_label_stat) {
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m) return;

    new_xyz += bs_idx * m * 3 + pt_idx * 3;
    xyz += bs_idx * n * 3;
    label_stat += bs_idx * n * nclass;
    new_label_stat += bs_idx * m * nclass + pt_idx * nclass;

    for(int i = 0; i < nclass; i++){
        new_label_stat[i] = 0;
    }

    float radius2 = radius * radius;
    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];

    for (int k = 0; k < n; ++k) {
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        if (d2 < radius2){
            for(int i = 0; i < nclass; i++){
                new_label_stat[i] += label_stat[k * nclass + i];
            }
        }
    }
}


void labelstat_ballrange_cuda_launcher_fast(int b, int n, int m, float radius, int nclass,
    const float *new_xyz, const float *xyz, const int *label_stat, int *new_label_stat, cudaStream_t stream) {
    // param new_xyz: (B, m, 3)
    // param xyz: (B, n, 3)
    // param idx: (B, m, nsample)

    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    labelstat_ballrange_cuda_kernel_fast<<<blocks, threads, 0, stream>>>(b, n, m, radius, nclass, new_xyz, xyz, label_stat, new_label_stat);
    // cudaDeviceSynchronize();  // for using printf in kernel function

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

// input: idx(b, m, nsample) label_stat(b, n, nclass)
// output: new_label_stat(b, m, nclass)
__global__ void labelstat_idx_cuda_kernel_fast(int b, int n, int m, int nsample, int nclass,
    const int *label_stat, const int *idx, int *new_label_stat) {
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m) return;

    idx += bs_idx * m * nsample + pt_idx * nsample;
    label_stat += bs_idx * n * nclass;
    new_label_stat += bs_idx * m * nclass + pt_idx * nclass;

    for(int i = 0; i < nclass; i++){
        new_label_stat[i] = 0;
    }

    for(int k = 0; k < nsample; k++){
        const int *label_stat_k = label_stat + idx[k] * nclass;
        for(int i = 0; i < nclass; i++){
            new_label_stat[i] += label_stat_k[i];
        }
    }
}


void labelstat_idx_cuda_launcher_fast(int b, int n, int m, int nsample, int nclass,
    const int *label_stat, const int *idx, int *new_label_stat, cudaStream_t stream) {
    // param new_xyz: (B, m, 3)
    // param xyz: (B, n, 3)
    // param idx: (B, m, nsample)

    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    labelstat_idx_cuda_kernel_fast<<<blocks, threads, 0, stream>>>(b, n, m, nsample, nclass, label_stat, idx, new_label_stat);
    // cudaDeviceSynchronize();  // for using printf in kernel function

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}