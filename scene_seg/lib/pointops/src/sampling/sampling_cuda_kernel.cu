#include "../cuda_utils.h"
#include "sampling_cuda_kernel.h"

// input: points(b, c, n) idx(b, m)
// output: out(b, c, m)
__global__ void gathering_forward_cuda_kernel(int b, int c, int n, int m, const float *points, const int *idx, float *out)
{
    for (int i = blockIdx.x; i < b; i += gridDim.x)
    {
        for (int l = blockIdx.y; l < c; l += gridDim.y)
        {
            for (int j = threadIdx.x; j < m; j += blockDim.x)
            {
                int a = idx[i * m + j];
                out[(i * c + l) * m + j] = points[(i * c + l) * n + a];
            }
        }
    }
}

// input: grad_out(b, c, m) idx(b, m)
// output: grad_points(b, c, n)
__global__ void gathering_backward_cuda_kernel(int b, int c, int n, int m, const float *grad_out, const int *idx, float *grad_points)
{
    for (int i = blockIdx.x; i < b; i += gridDim.x)
    {
        for (int l = blockIdx.y; l < c; l += gridDim.y)
        {
            for (int j = threadIdx.x; j < m; j += blockDim.x)
            {
                int a = idx[i * m + j];
                atomicAdd(grad_points + (i * c + l) * n + a, grad_out[(i * c + l) * m + j]);
            }
        }
    }
}

void gathering_forward_cuda_launcher(int b, int c, int n, int m, const float *points, const int *idx, float *out)
{
    gathering_forward_cuda_kernel<<<dim3(b, c, 1), opt_n_threads(m), 0>>>(b, c, n, m, points, idx, out);
}

void gathering_backward_cuda_launcher(int b, int c, int n, int m, const float *grad_out, const int *idx, float *grad_points)
{
    gathering_backward_cuda_kernel<<<dim3(b, c, 1), opt_n_threads(m), 0>>>(b, c, n, m, grad_out, idx, grad_points);
}

__device__ void __update(float *dists, int *dists_i,
			 int idx1, int idx2) {
    const float v1 = dists[idx1], v2 = dists[idx2];
    const int i1 = dists_i[idx1], i2 = dists_i[idx2];
    dists[idx1] = max(v1, v2);
    dists_i[idx1] = v2 > v1 ? i2 : i1;
}

// Input dataset: (b, n, 3), tmp: (b, n)
// Ouput idxs (b, m)
template <unsigned int block_size>
__global__ void furthestsampling_cuda_kernel(int b, int n, int m, const float *dataset, float *temp, int *idxs)
{
    if (m <= 0)
	    return;
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];

    int batch_index = blockIdx.x;
    dataset += batch_index * n * 3;
    temp += batch_index * n;
    idxs += batch_index * m;
    int tid = threadIdx.x;
    const int stride = block_size;
    int old = 0;
    if (threadIdx.x == 0)
	    idxs[0] = old;

    __syncthreads();
    for (int j = 1; j < m; j++)
    {
        int besti = 0;
        float best = -1;
        float x1 = dataset[old * 3 + 0];
        float y1 = dataset[old * 3 + 1];
        float z1 = dataset[old * 3 + 2];
        for (int k = tid; k < n; k += stride)
        {
            float x2, y2, z2;
            x2 = dataset[k * 3 + 0];
            y2 = dataset[k * 3 + 1];
            z2 = dataset[k * 3 + 2];
            //float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
            //if (mag <= 1e-3)
            //    continue;
            float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
            float d2 = min(d, temp[k]);
            temp[k] = d2;
            besti = d2 > best ? k : besti;
            best = d2 > best ? d2 : best;
        }
        dists[tid] = best;
        dists_i[tid] = besti;
        __syncthreads();

        if (block_size >= 1024) {
            if (tid < 512) {
            __update(dists, dists_i, tid, tid + 512);
            }
            __syncthreads();
        }
        if (block_size >= 512) {
            if (tid < 256) {
            __update(dists, dists_i, tid, tid + 256);
            }
            __syncthreads();
        }
        if (block_size >= 256) {
            if (tid < 128) {
            __update(dists, dists_i, tid, tid + 128);
            }
            __syncthreads();
        }
        if (block_size >= 128) {
            if (tid < 64) {
            __update(dists, dists_i, tid, tid + 64);
            }
            __syncthreads();
        }
        if (block_size >= 64) {
            if (tid < 32) {
            __update(dists, dists_i, tid, tid + 32);
            }
            __syncthreads();
        }
        if (block_size >= 32) {
            if (tid < 16) {
            __update(dists, dists_i, tid, tid + 16);
            }
            __syncthreads();
        }
        if (block_size >= 16) {
            if (tid < 8) {
            __update(dists, dists_i, tid, tid + 8);
            }
            __syncthreads();
        }
        if (block_size >= 8) {
            if (tid < 4) {
            __update(dists, dists_i, tid, tid + 4);
            }
            __syncthreads();
        }
        if (block_size >= 4) {
            if (tid < 2) {
            __update(dists, dists_i, tid, tid + 2);
            }
            __syncthreads();
        }
        if (block_size >= 2) {
            if (tid < 1) {
            __update(dists, dists_i, tid, tid + 1);
            }
            __syncthreads();
        }

        old = dists_i[0];
        if (tid == 0)
            idxs[j] = old;
    }
}

void furthestsampling_cuda_launcher(int b, int n, int m, const float *dataset, float *temp, int *idxs)
{   
	unsigned int n_threads = opt_n_threads(n);
	switch (n_threads) {
	    case 1024:
	        furthestsampling_cuda_kernel<1024><<<b, n_threads, 0>>>(b, n, m, dataset, temp, idxs);
	        break;
		case 512:
			furthestsampling_cuda_kernel<512><<<b, n_threads, 0>>>(b, n, m, dataset, temp, idxs);
			break;
    	case 256:
			furthestsampling_cuda_kernel<256><<<b, n_threads, 0>>>(b, n, m, dataset, temp, idxs);
			break;
    	case 128:
			furthestsampling_cuda_kernel<128><<<b, n_threads, 0>>>(b, n, m, dataset, temp, idxs);
			break;
    	case 64:
			furthestsampling_cuda_kernel<64><<<b, n_threads, 0>>>(b, n, m, dataset, temp, idxs);
			break;
    	case 32:
			furthestsampling_cuda_kernel<32><<<b, n_threads, 0>>>(b, n, m, dataset, temp, idxs);
			break;
    	case 16:
			furthestsampling_cuda_kernel<16><<<b, n_threads, 0>>>(b, n, m, dataset, temp, idxs);
			break;
    	case 8:
			furthestsampling_cuda_kernel<8><<<b, n_threads, 0>>>(b, n, m, dataset, temp, idxs);
			break;
	    case 4:
			furthestsampling_cuda_kernel<4><<<b, n_threads, 0>>>(b, n, m, dataset, temp, idxs);
			break;
	    case 2:
			furthestsampling_cuda_kernel<2><<<b, n_threads, 0>>>(b, n, m, dataset, temp, idxs);
			break;
	    case 1:
			furthestsampling_cuda_kernel<1><<<b, n_threads, 0>>>(b, n, m, dataset, temp, idxs);
			break;
	    default:
			furthestsampling_cuda_kernel<512><<<b, n_threads, 0>>>(b, n, m, dataset, temp, idxs);
	    }
}
