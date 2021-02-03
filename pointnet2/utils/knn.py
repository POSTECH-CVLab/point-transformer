import open3d as o3d
import torch
import torch.utils.dlpack


def kNN(query, dataset, k):
    '''
    inputs
        query: (B, N0, D) shaped torch gpu Tensor.
        dataset: (B, N1, D) shaped torch gpu Tensor.
        k: int
    
    outputs
        neighbors: (B, N0, k) shaped torch Tensor.
                   Each row is the indices of a neighboring points.
    '''
    assert query.is_cuda and dataset.is_cuda, "Input tensors should be gpu tensors."
    assert query.dim() == 3 and dataset.dim(
    ) == 3, "Input tensors should be 3D."
    assert query.shape[0] == dataset.shape[
        0], "Input tensors should have same batch size."
    assert query.shape[2] == dataset.shape[
        2], "Input tensors should have same dimension."

    B, N1, _ = dataset.shape

    query_o3d = o3d.core.Tensor.from_dlpack(
        torch.utils.dlpack.to_dlpack(query))
    dataset_o3d = o3d.core.Tensor.from_dlpack(
        torch.utils.dlpack.to_dlpack(dataset))

    indices = []
    for i in range(query_o3d.shape[0]):
        _query = query_o3d[i]
        _dataset = dataset_o3d[i]
        nns = o3d.core.nns.NearestNeighborSearch(_dataset)
        status = nns.knn_index()
        if not status:
            raise Exception("Index failed.")
        neighbors, _ = nns.knn_search(_query, k)
        neighbors += N1 * i
        indices.append(torch.utils.dlpack.from_dlpack(neighbors.to_dlpack()))

    indices = torch.cat(indices, dim=0)
    return indices
