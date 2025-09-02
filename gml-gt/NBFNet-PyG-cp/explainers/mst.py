from networkx.utils import UnionFind
import torch


def kruskal_mst_edges(weight, edges, edge_id):
    """
    Iterate over edge of a Kruskal's algorithm min/max spanning tree.
    Reference: https://networkx.org/documentation/stable/_modules/networkx/algorithms/tree/mst.html
    """
    # sort the edges based on the weights.
    sorted_weight, indices = torch.sort(weight, descending=True)
    sorted_edges = edges[:, indices]
    sorted_edge_id = edge_id[indices]
    subtrees = UnionFind()

    # for loop. Add the selected edges to the list.
    sel_mask = []
    sel_edge_id = []

    for w, edge, edge_id in zip(sorted_weight, sorted_edges.T, sorted_edge_id):
        u = edge[0].item()
        v = edge[1].item()
        if subtrees[u] != subtrees[v]:
            sel_mask.append(w)
            sel_edge_id.append(edge_id)
            subtrees.union(u, v)

    if len(sel_mask) != 0:
        sel_mask = torch.stack(sel_mask)
        sel_edge_id = torch.stack(sel_edge_id)
    else:
        sel_mask = torch.tensor([], device=weight.device, dtype=weight.dtype)
        sel_edge_id = torch.tensor([],device=weight.device, dtype=sorted_edge_id.dtype)

    return sel_mask, sel_edge_id
