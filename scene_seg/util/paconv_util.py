import torch


def weight_init(m):
    # print(m)
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


def get_graph_feature(x, k, idx):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    neighbor = x.view(batch_size * num_points, -1)[idx, :]

    neighbor = neighbor.view(batch_size, num_points, k, num_dims)

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((neighbor - x, neighbor), dim=3)  # (xj-xi, xj): b,n,k,2c

    return feature


def assign_score(score, point_input):
    B, N, K, m = score.size()
    score = score.view(B, N, K, 1, m)
    point_output = torch.matmul(score, point_input).view(B, N, K, -1)  # b,n,k,cout
    return point_output


def get_ed(x, y):
    ed = torch.norm(x - y, dim=-1).reshape(x.shape[0], 1)
    return ed


def assign_kernel_withoutk(in_feat, kernel, M):
    B, Cin, N0 = in_feat.size()
    in_feat_trans = in_feat.permute(0, 2, 1)
    out_feat_half1 = torch.matmul(in_feat_trans, kernel[:Cin]).view(B, N0, M, -1)  # b,n,m,o1
    out_feat_half2 = torch.matmul(in_feat_trans, kernel[Cin:]).view(B, N0, M, -1)  # b,n,m,o1
    if in_feat.size(1) % 2 != 0:
        out_feat_half_coord = torch.matmul(in_feat_trans[:, :, :3], kernel[Cin: Cin + 3]).view(B, N0, M, -1)  # b,n,m,o1
    else:
        out_feat_half_coord = torch.zeros_like(out_feat_half2)
    return out_feat_half1 + out_feat_half2, out_feat_half1 + out_feat_half_coord