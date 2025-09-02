import torch
from nbfnet import tasks
import copy

def index_edge(graph, pair):
    return torch.where((graph.T == pair).all(dim=1))[0]

def index_to_mask(index, size):
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask

def remove_target_edge(num_original_relation, data, h_index, t_index, r_index=None):
    # we remove training edges (we need to predict them at training time) from the edge index
    # think of it as a dynamic edge dropout
    h_index_ext = torch.cat([h_index, t_index], dim=-1)
    t_index_ext = torch.cat([t_index, h_index], dim=-1)
    inv_rels = r_index >= num_original_relation # create a mask where there are inv_rels

    # Compute the inverse relations using vectorized operations
    r_ext = r_index.clone()
    r_ext[inv_rels] -= num_original_relation
    r_ext[~inv_rels] += num_original_relation

    r_index_ext = torch.cat([r_index, r_ext], dim=-1)

    # we remove existing immediate edges between heads and tails in the batch with the given relation
    edge_index = torch.cat([data.edge_index, data.edge_type.unsqueeze(0)])
    # note that here we add relation types r_index_ext to the matching query
    easy_edge = torch.stack([h_index_ext, t_index_ext, r_index_ext]).flatten(1)
    index = tasks.edge_match(edge_index, easy_edge)[0]
    mask = ~index_to_mask(index, data.num_edges)

    data = copy.copy(data) # deep copy so you don't keep deleting the edges
    data.edge_index = data.edge_index[:, mask]
    data.edge_type = data.edge_type[mask]

    return data