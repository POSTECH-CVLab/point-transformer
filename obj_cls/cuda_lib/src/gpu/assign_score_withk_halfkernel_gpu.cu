#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "cuda_utils.h"
#include "utils.h"


// input: points(B,N,M,O), scores(B,N,K,M), idx(B,N,K)
// ouput: fout(B,O,N)
// algo: fout(b,i,k,j) = s(b,i,k,m)*p(b,i,k,m,j) =  s(b,i,k,m)*p(b,i(k),m,j)
//       i(k) = idx(b,i,k)
//      sum: fout(b,i,j) = fout(b,i,j) + s(b,i,k,m)*p(b,i,k,m,j)
//      avg: fout(b,i,j) = sum(fout(b,i,k,j)) / k
//      max: fout(b,i,j) = max(fout(b,i,k,j), sum(s(b,i,k,m)*p(b,i,k,m,j)))
// k,m : sequential
// b,n: parallel

const int SUM = 0;
const int AVG = 1;
const int MAX = 2;

#ifndef _CLOCK_T_DEFINED
typedef long clock_t;
#define _CLOCK_T_DEFINED
#endif

__global__ void assign_score_withk_halfkernel_forward_kernel(const int nthreads, const int B, const int N, const int M,
                                    const int K, const int O, const int aggregate,
                                    const float* points,
                                    const float* scores,
                                    const long* knn_idx,
                                    float* output) {

    // clock_t start, finish;
	// start = clock();s

    // ----- parallel loop for B, N and O ---------
    for (long i = blockIdx.x * blockDim.x + threadIdx.x; i < nthreads; i += blockDim.x * gridDim.x) {
        // ----- loop for K ---------
        for (int k = 0; k < K; k++) {
            int b = (int)(i / (O * N));
            int n = (int)(i % (O * N) / O);
            int o = (int)(i % O);
            float tmp = 0;
            // ------- loop for M ----------
            for (int m = 0; m < M; m++) {
                int kn = (int) knn_idx[b*K*N + n*K + k];
                assert (kn < N);
                assert (o < O);
                assert (n < N);

                if (aggregate == SUM) {
                    // feature concat
                    // output[b*N*O + o*N + n] += 2 * points[b*N*M*O + kn*M*O + m*O + o] * scores[b*N*K*M + n*K*M + k*M + m];
                    // output[b*N*O + o*N + n] -= points[b*N*M*O + n*M*O + m*O + o] * scores[b*N*K*M + n*K*M + k*M + m];
                    atomicAdd(output + b*N*O + o*N + n,
                        2 * points[b*N*M*O + kn*M*O + m*O + o] * scores[b*N*K*M + n*K*M + k*M + m]
                            - points[b*N*M*O + n*M*O + m*O + o] * scores[b*N*K*M + n*K*M + k*M + m]);
                }
                else if (aggregate == AVG) {
                     atomicAdd(output + b*N*O + o*N + n,
                        (2 * points[b*N*M*O + kn*M*O + m*O + o] * scores[b*N*K*M + n*K*M + k*M + m]
                            - points[b*N*M*O + n*M*O + m*O + o] * scores[b*N*K*M + n*K*M + k*M + m]) / K);
                }
                else if (aggregate == MAX) {
                    atomicAdd(&tmp,
                        2 * points[b*N*M*O + kn*M*O + m*O + o] * scores[b*N*K*M + n*K*M + k*M + m]
                            - points[b*N*M*O + n*M*O + m*O + o] * scores[b*N*K*M + n*K*M + k*M + m]);
                }
            }

            if (aggregate == MAX) {
                output[b*N*O + o*N + n] = output[b*N*O + o*N + n] > tmp ? output[b*N*O + o*N + n] : tmp;
            }
        }
    }

    // finish = clock();
	// printf("assign socre forward time：blockid %d, %f\n", batch_idx, (double)(finish - start)/10000.0);
}


__global__ void assign_score_withk_halfkernel_backward_points_kernel(const int nthreads, const int B, const int N, const int M,
                                    const int K, const int O, const int aggregate,
                                    const float* grad_out,
                                    const float* points,
                                    const float* scores,
                                    const long* knn_idx,
                                    float* grad_points) {

    // clock_t start, finish;
	// start = clock();

    // ----- parallel loop for M, O ---------
    for (long i = blockIdx.x * blockDim.x + threadIdx.x; i < nthreads; i += blockDim.x * gridDim.x) {
        int b = (int)(i / (M * O));
        int m = (int)(i % (M * O) / O);
        int o = (int)(i % O);

        // ----- loop for N,K ---------
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                int kn = knn_idx[b*N*K + n*K + k];
                atomicAdd(grad_points + b*N*M*O + kn*M*O + m*O + o,
                    2 * scores[b*N*K*M + n*K*M + k*M + m] * grad_out[b*O*N + o*N + n]);
                atomicAdd(grad_points + b*N*M*O + n*M*O + m*O + o,
                    - scores[b*N*K*M + n*K*M + k*M + m] * grad_out[b*O*N + o*N + n]);
                // grad_points[b*N*M*O + kn*M*O + m*O + o] += 2 * scores[b*N*K*M + n*K*M + k*M + m] * grad_out[b*O*N + o*N + n];
                // grad_points[b*N*M*O + n*M*O + m*O + o] -= scores[b*N*K*M + n*K*M + k*M + m] * grad_out[b*O*N + o*N + n];
             }
        }

    }
    // finish = clock();
	// printf("assign socre backward time 1：blockid %d, %f\n", batch_idx, (double)(finish - start)/10000.0);

}


__global__ void assign_score_withk_halfkernel_backward_scores_kernel(const int nthreads, const int B, const int N, const int M,
                                    const int K, const int O, const int aggregate,
                                    const float* grad_out,
                                    const float* points,
                                    const float* scores,
                                    const long* knn_idx,
                                    float* grad_scores) {

    // clock_t start, finish;
	// start = clock();

    // ----- parallel loop for N, K, M ---------
    for (long i = blockIdx.x * blockDim.x + threadIdx.x; i < nthreads; i += blockDim.x * gridDim.x) {
    // for (int i = index; i < N*K*M; i += stride) {
        int b = (int)(i / (N * M * K));
        int n = (int)(i % (N * M * K) / M / K);
        int k = (int)(i % (M * K) / M);
        int m = (int)(i % M);
        int kn = knn_idx[b*N*K + n*K + k];

        for(int o = 0; o < O; o++) {
            atomicAdd(grad_scores + b*N*K*M + n*K*M + k*M + m,
                (2 * points[b*N*M*O + kn*M*O + m*O + o]
                    - points[b*N*M*O + n*M*O + m*O + o])* grad_out[b*O*N + o*N + n]);
            // grad_scores[b*N*K*M + n*K*M + k*M + m] += (2 * points[b*N*M*O + kn*M*O + m*O + o] - points[b*N*M*O + n*M*O + m*O + o])* grad_out[b*O*N + o*N + n];
        }
    }

    // finish = clock();
	// printf("assign socre backward time 2：blockid %d, %f\n", batch_idx, (double)(finish - start)/10000.0);
}


void assign_score_withk_halfkernel_forward_kernel_wrapper(int B, int N, int M, int K, int O, int aggregate,
                                                                     const at::Tensor& points,
                                                                     const at::Tensor& scores,
                                                                     const at::Tensor& knn_idx,
                                                                     at::Tensor& output) {
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(scores);
    CHECK_CONTIGUOUS(knn_idx);
    CHECK_CONTIGUOUS(output);

    const float* points_data = points.data_ptr<float>();
    const float* scores_data = scores.data_ptr<float>();
    const long* knn_idx_data = knn_idx.data_ptr<long>();
    float* output_data = output.data_ptr<float>();

    int nthreads = B * N * O; // * K * M;

    assign_score_withk_halfkernel_forward_kernel<<<nthreads, 512>>>(
        nthreads, B, N, M, K, O, aggregate, points_data, scores_data, knn_idx_data, output_data);

    CUDA_CHECK_ERRORS();

}


void assign_score_withk_halfkernel_backward_kernel_wrapper(int B, int N, int M, int K, int O, int aggregate,
                                                                     const at::Tensor& grad_out,
                                                                     const at::Tensor& points,
                                                                     const at::Tensor& scores,
                                                                     const at::Tensor& knn_idx,
                                                                     at::Tensor& grad_points,
                                                                     at::Tensor& grad_scores) {

    CHECK_CONTIGUOUS(grad_out);
    CHECK_CONTIGUOUS(scores);
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(knn_idx);
    CHECK_CONTIGUOUS(grad_scores);
    CHECK_CONTIGUOUS(grad_points);

    const float* grad_out_data = grad_out.data_ptr<float>();
    const float* points_data = points.data_ptr<float>();
    const float* scores_data = scores.data_ptr<float>();
    const long* knn_idx_data = knn_idx.data_ptr<long>();
    float* grad_points_data = grad_points.data_ptr<float>();
    float* grad_scores_data = grad_scores.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int nthreads_1 = B * M * O;
    int nthreads_2 = B * N * K * M;

    assign_score_withk_halfkernel_backward_points_kernel<<<nthreads_1, 512>>>(
        nthreads_1, B, N, M, K, O, aggregate, grad_out_data, points_data, scores_data, knn_idx_data, grad_points_data);
    assign_score_withk_halfkernel_backward_scores_kernel<<<nthreads_2, 512>>>(
        nthreads_2, B, N, M, K, O, aggregate, grad_out_data, points_data, scores_data, knn_idx_data, grad_scores_data);

    CUDA_CHECK_ERRORS();

}
