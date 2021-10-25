import torch
from torch.autograd import Function

from .. import src


class AssignScoreWithKHalfKernel(Function):
    @staticmethod
    def forward(ctx, scores, points, knn_idx, aggregate) : # -> torch.Tensor:
        """
        :param ctx
        :param scores: (B, N, K, M)
        :param points: (B, N, M, O)
        :param knn_idx: (B, N, K)
        :param aggregate:
        :return: output: (B, O, N)
        """

        agg = {'sum': 0, 'avg': 1, 'max': 2}

        B, N, M, O = points.size()
        K = scores.size(2)

        output = torch.zeros([B, O, N], dtype=points.dtype, device=points.device)
        output = output.contiguous()

        src.gpu.assign_score_withk_halfkernel_forward_cuda(B, N, M, K, O, agg[aggregate],
                                                           points.contiguous(), scores.contiguous(),
                                                           knn_idx.contiguous(), output)

        ctx.save_for_backward(output, points, scores, knn_idx)
        ctx.agg = agg[aggregate]

        return output

    @staticmethod
    def backward(ctx, grad_out):
        """

        :param ctx:
        :param grad_out: (B, O, N) tensor with gradients of ouputs
        :return: grad_scores: (B, N, K, M) tensor with gradients of scores
        :return: grad_points: (B, N, M, O) tensor with gradients of point features
        """
        output, points, scores, knn_idx = ctx.saved_tensors

        agg = ctx.agg

        B, N, M, O = points.size()
        K = scores.size(2)

        grad_points = torch.zeros_like(points, dtype=points.dtype, device=points.device).contiguous()
        grad_scores = torch.zeros_like(scores, dtype=scores.dtype, device=scores.device).contiguous()

        src.gpu.assign_score_withk_halfkernel_backward_cuda(B, N, M, K, O, agg, grad_out.contiguous(),
                                                            points.contiguous(), scores.contiguous(), knn_idx.contiguous(),
                                                            grad_points, grad_scores)

        return grad_scores, grad_points, None, None


assign_score_withk_halfkernel = AssignScoreWithKHalfKernel.apply


class AssignScoreWithK(Function):
    @staticmethod
    def forward(ctx, scores, points, centers, knn_idx, aggregate):  # -> torch.Tensor:
        """
        :param ctx
        :param scores: (B, N, K, M)
        :param points: (B, N, M, O)
        :param centers: (B, N, M, O)
        :param knn_idx: (B, N, K)
        :param aggregate:
        :return: output: (B, O, N)
        """

        agg = {'sum': 0, 'avg': 1, 'max': 2}

        B, N, M, O = points.size()
        K = scores.size(2)

        output = torch.zeros([B, O, N], dtype=points.dtype, device=points.device)
        output = output.contiguous()

        src.gpu.assign_score_withk_forward_cuda(B, N, M, K, O, agg[aggregate],
                                                points.contiguous(), centers.contiguous(),
                                                scores.contiguous(), knn_idx.contiguous(),
                                                output)

        ctx.save_for_backward(output, points, centers, scores, knn_idx)
        ctx.agg = agg[aggregate]

        return output

    @staticmethod
    def backward(ctx, grad_out):
        """

        :param ctx:
        :param grad_out: (B, O, N) tensor with gradients of ouputs
        :return: grad_scores: (B, N, K, M) tensor with gradients of scores
        :return: grad_points: (B, N, M, O) tensor with gradients of point features
        :return: grad_centers: (B, N, M, O) tensor with gradients of center point features
        """
        output, points, centers, scores, knn_idx = ctx.saved_tensors

        agg = ctx.agg

        B, N, M, O = points.size()
        K = scores.size(2)

        grad_points = torch.zeros_like(points, dtype=points.dtype, device=points.device).contiguous()
        grad_centers = torch.zeros_like(centers, dtype=points.dtype, device=points.device).contiguous()
        grad_scores = torch.zeros_like(scores, dtype=scores.dtype, device=scores.device).contiguous()

        src.gpu.assign_score_withk_backward_cuda(B, N, M, K, O, agg, grad_out.contiguous(),
                                                 points.contiguous(), centers.contiguous(),
                                                 scores.contiguous(), knn_idx.contiguous(),
                                                 grad_points, grad_centers, grad_scores)

        return grad_scores, grad_points, grad_centers, None, None, None


assign_score_withk = AssignScoreWithK.apply
