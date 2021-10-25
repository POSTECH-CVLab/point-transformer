#include "../cuda_utils.h"
#include "knnquery_cuda_kernel.h"

// input: xyz (b, n, 3) new_xyz (b, m, 3)
// output: idx (b, m, nsample) dist2 (b, m, nsample)
__global__ void knnquery_cuda_kernel(int b, int n, int m, int nsample, const float *__restrict__ xyz, const float *__restrict__ new_xyz, int *__restrict__ idx, float *__restrict__ dist2) {
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m) return;

    new_xyz += bs_idx * m * 3 + pt_idx * 3;
    xyz += bs_idx * n * 3;
    idx += bs_idx * m * nsample + pt_idx * nsample;

    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];

    //double* best = new double[nsample];
    //int* besti = new int[nsample];
    double best[200];
    int besti[200];
    for(int i = 0; i < nsample; i++){
        best[i] = 1e40;
        besti[i] = 0;
    }
    for(int k = 0; k < n; k++){
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        for(int j = 0; j < nsample; j++){
            if(d2 < best[j]){
                for(int i = nsample - 1; i > j; i--){
                    best[i] = best[i - 1];
                    besti[i] = besti[i - 1];
                }
                best[j] = d2;
                besti[j] = k;
                break;
            }
        }
    }
    for(int i = 0; i < nsample; i++){
        idx[i] = besti[i];
        dist2[i] = best[i];
    }
    //delete []best;
    //delete []besti;
}


void knnquery_cuda_launcher(int b, int n, int m, int nsample, const float *xyz, const float *new_xyz, int *idx, float *dist2, cudaStream_t stream) {
    // param new_xyz: (B, m, 3)
    // param xyz: (B, n, 3)
    // param idx: (B, m, nsample)

    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    
    // fprintf('%d, %d', blocks, threads);
    knnquery_cuda_kernel<<<blocks, threads, 0, stream>>>(b, n, m, nsample, xyz, new_xyz, idx, dist2);
    // cudaDeviceSynchronize();  // for using printf in kernel function

    // err = cudaGetLastError();
    // if (cudaSuccess != err) {
    //     fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    //     exit(-1);
    // }
}