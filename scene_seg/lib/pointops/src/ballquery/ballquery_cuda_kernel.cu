#include "../cuda_utils.h"
#include "ballquery_cuda_kernel.h"

// input: new_xyz(b, m, 3) xyz(b, n, 3)
// output: idx(b, m, nsample)
__global__ void ballquery_cuda_kernel(int b, int n, int m, float radius, int nsample, const float *new_xyz, const float *xyz, int *idx)
{
    int batch_index = blockIdx.x;
    xyz += batch_index * n * 3;
    new_xyz += batch_index * m * 3;
    idx += m * nsample * batch_index;
    int index = threadIdx.x;
    int stride = blockDim.x;

    float radius2 = radius * radius;
    for (int j = index; j < m; j += stride)
    {
        float new_x = new_xyz[j * 3 + 0];
        float new_y = new_xyz[j * 3 + 1];
        float new_z = new_xyz[j * 3 + 2];
        for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k)
        {
            float x = xyz[k * 3 + 0];
            float y = xyz[k * 3 + 1];
            float z = xyz[k * 3 + 2];
            float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
            if (d2 < radius2)
            {
                if (cnt == 0)
                {
                    for (int l = 0; l < nsample; ++l)
                        idx[j * nsample + l] = k;
                }
                idx[j * nsample + cnt] = k;
                ++cnt;
            }
        }
    }
}

void ballquery_cuda_launcher(int b, int n, int m, float radius, int nsample, const float *new_xyz, const float *xyz, int *idx)
{
    ballquery_cuda_kernel<<<b, opt_n_threads(m), 0>>>(b, n, m, radius, nsample, new_xyz, xyz, idx);
}


__global__ void ballquery_cuda_kernel_fast(int b, int n, int m, float radius, int nsample, const float *__restrict__ new_xyz, const float *__restrict__ xyz, int *__restrict__ idx) {
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m) return;

    new_xyz += bs_idx * m * 3 + pt_idx * 3;
    xyz += bs_idx * n * 3;
    idx += bs_idx * m * nsample + pt_idx * nsample;

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


void ballquery_cuda_launcher_fast(int b, int n, int m, float radius, int nsample, const float *new_xyz, const float *xyz, int *idx, cudaStream_t stream) {
    // param new_xyz: (B, m, 3)
    // param xyz: (B, n, 3)
    // param idx: (B, m, nsample)

    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    ballquery_cuda_kernel_fast<<<blocks, threads, 0, stream>>>(b, n, m, radius, nsample, new_xyz, xyz, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
