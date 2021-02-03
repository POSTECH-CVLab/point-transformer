import open3d as o3d
import torch
import torch.utils.dlpack

def kNN(query, dataset, k):
    '''
    inputs
        query: (n0, dim) shaped torch gpu Tensor.
        dataset: (n1, dim) shaped torch gpu Tensor.
        k: int
    
    outputs
        neighbors: (n0, k) shaped torch Tensor.
                   Each row is the indices of a neighboring points.
    '''
    assert query.is_cuda and dataset.is_cuda, "Input tensors should be gpu tensors."
    assert query.dim() == 2 and dataset.dim(
    ) == 2, "Input tensors should be 2D."
    assert query.shape[1] == dataset.shape[
        1], "Input tensors should have same dimension."

    query_o3d = o3d.core.Tensor.from_dlpack(
        torch.utils.dlpack.to_dlpack(query))
    dataset_o3d = o3d.core.Tensor.from_dlpack(
        torch.utils.dlpack.to_dlpack(dataset))

    nns = o3d.core.nns.NearestNeighborSearch(dataset_o3d)
    status = nns.knn_index()
    if not status:
        raise Exception("Index failed.")
    indices, _ = nns.knn_search(query_o3d, k)
    indices = torch.utils.dlpack.from_dlpack(indices.to_dlpack())
    return indices
