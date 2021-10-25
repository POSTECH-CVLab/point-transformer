//
// Created by Runyu Ding on 2020/8/12.
//

#ifndef _OPERATOR_H
#define _OPERATOR_H

#include <torch/torch.h>

void assign_score_withk_halfkernel_forward_kernel_wrapper(int B, int N, int M, int K, int O, int aggregate,
                                 const at::Tensor& points,
                                 const at::Tensor& scores,
                                 const at::Tensor& knn_idx,
                                 at::Tensor& output);

void assign_score_withk_halfkernel_backward_kernel_wrapper(int B, int N, int M, int K, int O, int aggregate,
                                                                     const at::Tensor& grad_out,
                                                                     const at::Tensor& points,
                                                                     const at::Tensor& scores,
                                                                     const at::Tensor& knn_idx,
                                                                     at::Tensor& grad_points,
                                                                     at::Tensor& grad_scores);

void assign_score_withk_forward_kernel_wrapper(int B, int N, int M, int K, int O, int aggregate,
                                 const at::Tensor& points,
                                 const at::Tensor& centers,
                                 const at::Tensor& scores,
                                 const at::Tensor& knn_idx,
                                 at::Tensor& output);

void assign_score_withk_backward_kernel_wrapper(int B, int N, int M, int K, int O, int aggregate,
                                 const at::Tensor& grad_out,
                                 const at::Tensor& points,
                                 const at::Tensor& centers,
                                 const at::Tensor& scores,
                                 const at::Tensor& knn_idx,
                                 at::Tensor& grad_points,
                                 at::Tensor& grad_centers,
                                 at::Tensor& grad_scores);


#endif