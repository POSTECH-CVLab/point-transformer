#include "../cuda_utils.h"
#include "interpolation_cuda_kernel.h"

// input: unknown(b, n, 3) known(b, m, 3)
// output: dist2(b, n, 3), idx(b, n, 3)
__global__ void nearestneighbor_cuda_kernel(int b, int n, int m, const float *unknown, const float *known, float *dist2, int *idx)
{
    int batch_index = blockIdx.x;
    unknown += batch_index * n * 3;
    known += batch_index * m * 3;
    dist2 += batch_index * n * 3;
    idx += batch_index * n * 3;

    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int j = index; j < n; j += stride)
    {
		float ux = unknown[j * 3 + 0];
		float uy = unknown[j * 3 + 1];
		float uz = unknown[j * 3 + 2];

		double best1 = 1e40, best2 = 1e40, best3 = 1e40;
		int besti1 = 0, besti2 = 0, besti3 = 0;
		for (int k = 0; k < m; ++k)
		{
		    float x = known[k * 3 + 0];
		    float y = known[k * 3 + 1];
		    float z = known[k * 3 + 2];
		    float d =
			(ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
		    if (d < best1)
		    {
				best3 = best2;
				besti3 = besti2;
				best2 = best1;
				besti2 = besti1;
				best1 = d;
				besti1 = k;
			}
			else if (d < best2)
			{
				best3 = best2;
				besti3 = besti2;
				best2 = d;
				besti2 = k;
			}
			else if (d < best3)
			{
				best3 = d;
				besti3 = k;
		    }
		}
		dist2[j * 3 + 0] = best1;
		dist2[j * 3 + 1] = best2;
		dist2[j * 3 + 2] = best3;
		idx[j * 3 + 0] = besti1;
		idx[j * 3 + 1] = besti2;
		idx[j * 3 + 2] = besti3;
    }
}

// input: points(b, c, m), idx(b, n, 3), weight(b, n, 3)
// output: out(b, c, n)
__global__ void interpolation_forward_cuda_kernel(int b, int c, int m, int n, const float *points, const int *idx, const float *weight, float *out)
{
    int batch_index = blockIdx.x;
    points += batch_index * m * c;
    idx += batch_index * n * 3;
    weight += batch_index * n * 3;
    out += batch_index * n * c;

    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    const int stride = blockDim.y * blockDim.x;
    for (int i = index; i < c * n; i += stride)
    {
		const int l = i / n;
		const int j = i % n;
		float w1 = weight[j * 3 + 0];
		float w2 = weight[j * 3 + 1];
		float w3 = weight[j * 3 + 2];
		int i1 = idx[j * 3 + 0];
		int i2 = idx[j * 3 + 1];
		int i3 = idx[j * 3 + 2];
		out[i] = points[l * m + i1] * w1 + points[l * m + i2] * w2 + points[l * m + i3] * w3;
    }
}

// input: grad_out(b, c, n), idx(b, n, 3), weight(b, n, 3)
// output: grad_points(b, c, m)
__global__ void interpolation_backward_cuda_kernel( int b, int c, int n, int m, const float *grad_out, const int *idx, const float *weight, float *grad_points)
{
    int batch_index = blockIdx.x;
    grad_out += batch_index * n * c;
    idx += batch_index * n * 3;
    weight += batch_index * n * 3;
    grad_points += batch_index * m * c;

    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    const int stride = blockDim.y * blockDim.x;
    for (int i = index; i < c * n; i += stride)
    {
		const int l = i / n;
		const int j = i % n;
		float w1 = weight[j * 3 + 0];
		float w2 = weight[j * 3 + 1];
		float w3 = weight[j * 3 + 2];
		int i1 = idx[j * 3 + 0];
		int i2 = idx[j * 3 + 1];
		int i3 = idx[j * 3 + 2];
		atomicAdd(grad_points + l * m + i1, grad_out[i] * w1);
		atomicAdd(grad_points + l * m + i2, grad_out[i] * w2);
		atomicAdd(grad_points + l * m + i3, grad_out[i] * w3);
    }
}

void nearestneighbor_cuda_launcher(int b, int n, int m, const float *unknown, const float *known, float *dist2, int *idx)
{
    nearestneighbor_cuda_kernel<<<b, opt_n_threads(n), 0>>>(b, n, m, unknown, known, dist2, idx);
}

void interpolation_forward_cuda_launcher(int b, int c, int m, int n, const float *points, const int *idx, const float *weight, float *out)
{
    interpolation_forward_cuda_kernel<<<b, opt_block_config(n, c), 0>>>(b, c, m, n, points, idx, weight, out);
}

void interpolation_backward_cuda_launcher(int b, int n, int c, int m, const float *grad_out, const int *idx, const float *weight, float *grad_points)
{
    interpolation_backward_cuda_kernel<<<b, opt_block_config(n, c), 0>>>(b, n, c, m, grad_out, idx, weight, grad_points);
}


// input: unknown(b, n, 3) known(b, m, 3)
// output: dist2(b, n, 3), idx(b, n, 3)
__global__ void nearestneighbor_cuda_kernel_fast(int b, int n, int m, const float *__restrict__ unknown, const float *__restrict__ known, float *__restrict__ dist2, int *__restrict__ idx) {

    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= n) return;

    unknown += bs_idx * n * 3 + pt_idx * 3;
    known += bs_idx * m * 3;
    dist2 += bs_idx * n * 3 + pt_idx * 3;
    idx += bs_idx * n * 3 + pt_idx * 3;

    float ux = unknown[0];
    float uy = unknown[1];
    float uz = unknown[2];

    double best1 = 1e40, best2 = 1e40, best3 = 1e40;
    int besti1 = 0, besti2 = 0, besti3 = 0;
    for (int k = 0; k < m; ++k) {
        float x = known[k * 3 + 0];
        float y = known[k * 3 + 1];
        float z = known[k * 3 + 2];
        float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
        if (d < best1) {
            best3 = best2; besti3 = besti2;
            best2 = best1; besti2 = besti1;
            best1 = d; besti1 = k;
        }
        else if (d < best2) {
            best3 = best2; besti3 = besti2;
            best2 = d; besti2 = k;
        }
        else if (d < best3) {
            best3 = d; besti3 = k;
        }
    }
    dist2[0] = best1;
    dist2[1] = best2;
    dist2[2] = best3;

    idx[0] = besti1;
    idx[1] = besti2;
    idx[2] = besti3;
}


// input: points(b, c, m), idx(b, n, 3), weight(b, n, 3)
// output: out(b, c, n)
__global__ void interpolation_forward_cuda_kernel_fast(int b, int c, int m, int n, const float *__restrict__ points, const int *__restrict__ idx, const float *__restrict__ weight, float *__restrict__ out) {

    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bs_idx >= b || c_idx >= c || pt_idx >= n) return;

    weight += bs_idx * n * 3 + pt_idx * 3;
    points += bs_idx * c * m + c_idx * m;
    idx += bs_idx * n * 3 + pt_idx * 3;
    out += bs_idx * c * n + c_idx * n;

    out[pt_idx] = weight[0] * points[idx[0]] + weight[1] * points[idx[1]] + weight[2] * points[idx[2]];
}


void nearestneighbor_cuda_launcher_fast(int b, int n, int m, const float *unknown, const float *known, float *dist2, int *idx)
{
    cudaError_t err;

    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    nearestneighbor_cuda_kernel_fast<<<blocks, threads, 0>>>(b, n, m, unknown, known, dist2, idx);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void interpolation_forward_cuda_launcher_fast(int b, int c, int m, int n, const float *points, const int *idx, const float *weight, float *out) {

    cudaError_t err;

    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    interpolation_forward_cuda_kernel_fast<<<blocks, threads, 0>>>(b, c, m, n, points, idx, weight, out);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n",
        cudaGetErrorString(err));
        exit(-1);
    }
}
