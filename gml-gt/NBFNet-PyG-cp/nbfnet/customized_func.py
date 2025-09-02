from typing import Tuple, Optional
import torch
from torch_scatter import scatter_sum, scatter_mul, scatter_min, scatter_max
from torch_scatter.utils import broadcast
from torch_geometric.utils.num_nodes import maybe_num_nodes


def scatter_mean(src: torch.Tensor, index: torch.Tensor, preserved_edges: torch.Tensor,
                 dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    # index_dim = dim
    # if index_dim < 0:
    #     index_dim = index_dim + src.dim()
    # if index.dim() <= index_dim:
    #     index_dim = index.dim() - 1

    # ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    # count = scatter_sum(ones, index, index_dim, None, dim_size)
    count = scatter_sum(preserved_edges, index, dim, None, dim_size) # count for each triple in batch, & for each node, how many edges they appear in
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode='floor')
    return out

def customized_scatter(src: torch.Tensor, index: torch.Tensor, preserved_edges: torch.Tensor,
                       dim: int = -1,
                       out: Optional[torch.Tensor] = None,
                       dim_size: Optional[int] = None,
                       reduce: str = "sum") -> torch.Tensor:
    if reduce == 'sum' or reduce == 'add':
        return scatter_sum(src, index, dim, out, dim_size)
    if reduce == 'mul': 
        # the input has been multipled by 0 for the places with dropped edges
        # add 1 to the elements for dropped edges s.t. we don't multiply by 0.
        dropped_edges = preserved_edges==0
        dropped_edges = broadcast(dropped_edges, src, dim)
        src = src+dropped_edges
        return scatter_mul(src, index, dim, out, dim_size) # this also needs to change
    elif reduce == 'mean':
        # adjust the count of the nodes
        return scatter_mean(src, index, preserved_edges, dim, out, dim_size) # this needs to change
    elif reduce == 'min':
        # the input is 0 for the places with dropped edges
        # make these inf s.t. it doesn't affect min operation. even if all the edges for node u is dropped, it still has the self-loop so inf will never be min
        dropped_edges = preserved_edges==0
        dropped_edges = broadcast(dropped_edges, src, dim)
        src[dropped_edges] = float('inf')
        return scatter_min(src, index, dim, out, dim_size)[0]
    elif reduce == 'max':
        # the input is 0 for the places with dropped edges
        # make these -inf s.t. it doesn't affect min operation. even if all the edges for node u is dropped, it still has the self-loop so -inf will never be max
        dropped_edges = preserved_edges==0
        dropped_edges = broadcast(dropped_edges, src, dim)
        src[dropped_edges] = float('-inf') 
        return scatter_max(src, index, dim, out, dim_size)[0]
    else:
        raise ValueError

def customized_degree(index: torch.Tensor, preserved_edges: torch.Tensor,
                      dim: int = -1,
                      num_nodes: Optional[int] = None,
                      dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    N = maybe_num_nodes(index, num_nodes)
    out = scatter_sum(preserved_edges, index, dim, None, N).squeeze()
    return out
