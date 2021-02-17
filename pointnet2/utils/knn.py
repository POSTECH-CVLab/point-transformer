import open3d as o3d
import torch
import torch.utils.dlpack


def kNN(query, dataset, k):
    """
    inputs
        query: (B, N0, D) shaped torch gpu Tensor.
        dataset: (B, N1, D) shaped torch gpu Tensor.
        k: int

    outputs
        neighbors: (B * N0, k) shaped torch Tensor.
                   Each row is the indices of a neighboring points.
                   It is flattened along batch dimension.
    """
    assert query.is_cuda and dataset.is_cuda, "Input tensors should be gpu tensors."
    assert query.dim() == 3 and dataset.dim() == 3, "Input tensors should be 3D."
    assert (
        query.shape[0] == dataset.shape[0]
    ), "Input tensors should have same batch size."
    assert (
        query.shape[2] == dataset.shape[2]
    ), "Input tensors should have same dimension."

    B, N1, _ = dataset.shape

    query_o3d = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(query))
    dataset_o3d = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(dataset))

    indices = []
    for i in range(query_o3d.shape[0]):
        _query = query_o3d[i]
        _dataset = dataset_o3d[i]
        nns = o3d.core.nns.NearestNeighborSearch(_dataset)
        status = nns.knn_index()
        if not status:
            raise Exception("Index failed.")
        neighbors, _ = nns.knn_search(_query, k)
        # calculate prefix sum of indices
        # neighbors += N1 * i
        indices.append(torch.utils.dlpack.from_dlpack(neighbors.to_dlpack()))

    # flatten indices
    indices = torch.stack(indices)
    return indices


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def kNN_torch(query, dataset, k):
    """
    inputs
        query: (B, N0, D) shaped torch gpu Tensor.
        dataset: (B, N1, D) shaped torch gpu Tensor.
        k: int

    outputs
        neighbors: (B * N0, k) shaped torch Tensor.
                   Each row is the indices of a neighboring points.
                   It is flattened along batch dimension.
    """
    assert query.is_cuda and dataset.is_cuda, "Input tensors should be gpu tensors."
    assert query.dim() == 3 and dataset.dim() == 3, "Input tensors should be 3D."
    assert (
        query.shape[0] == dataset.shape[0]
    ), "Input tensors should have same batch size."
    assert (
        query.shape[2] == dataset.shape[2]
    ), "Input tensors should have same dimension."

    dists = square_distance(query, dataset)  # dists: [B, N0, N1]
    neighbors = dists.argsort()[:, :, :k]  # neighbors: [B, N0, k]
    torch.cuda.empty_cache()
    return neighbors


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)