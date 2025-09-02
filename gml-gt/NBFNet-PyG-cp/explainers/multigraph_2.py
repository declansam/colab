def prepare_edges_for_rw(self, data, mask):
    """
    1. Knowledge graph can be multigraph, convert this to digraph (only one edge in u, v) by aggregating the weights for each relation.
    2. The central nodes / nodes in masked edges can be a singleton. Add self loop so the power method works.
    Aggr options: max, sum, mean
    """
    # each edge in every subgraph gets a unique id
    offset = data.edge_batch * data.num_nodes
    unique_edges = data.edge_index + offset

    # for each original edge, inverse_indices is the corresponding index in unique edges
    unique_edges, inverse_indices = torch.unique(
        unique_edges, return_inverse=True, dim=1
    )

    num_relations = torch.max(data.edge_type) + 1
    # for each unique edge, multi_edge_weights will get the score of the mask for each available edge type
    multi_edge_weights = torch.zeros(
        (unique_edges.size(1), num_relations), device=unique_edges.device
    )
    multi_edge_weights[inverse_indices, data.edge_type] = mask
    if self.adj_aggr == "max":
        # for each unique edge, take the relation with highest weight (lowest negative weight)
        mask, _ = torch.max(multi_edge_weights, dim=-1)
    elif self.adj_aggr == "mean":
        dense_edge_filter = torch.zeros_like(multi_edge_weights)
        dense_edge_filter[inverse_indices, data.edge_type] = 1
        norm = dense_edge_filter.sum(dim=-1)
        mask = torch.sum(multi_edge_weights, dim=-1) / norm
    elif self.adj_aggr == "sum":
        mask = torch.sum(multi_edge_weights, dim=-1)
    else:
        raise ValueError(f"Unknown adj_aggr type: {self.adj_aggr}")

    batch_size = data.edge_filter.size(0)
    # check if there are any singleton nodes
    # if so, add self loops so it doesn't mess up the random walk
    indices = (
        torch.arange(data.max_num_nodes, device=data.edge_index.device)
        .unsqueeze(0)
        .repeat(batch_size, 1)
    )
    valid_nodes = indices < data.subgraph_num_nodes.unsqueeze(1)
    node_ids = indices[valid_nodes]
    node_ids = node_ids + data.node_batch * data.num_nodes
    singleton_mask = ~torch.isin(node_ids, unique_edges)
    singleton = node_ids[singleton_mask]
    singleton_batch = data.node_batch[singleton_mask]
    self_loops = torch.stack([singleton, singleton])
    singleton_mask = torch.ones_like(singleton_batch)
    mask = torch.cat((mask, singleton_mask))

    # map the unique edges back to its original
    # get the edge_batch for the unique edges
    unique_edge_batch = torch.zeros(
        (unique_edges.size(1), batch_size),
        device=unique_edges.device,
        dtype=torch.long,
    )
    unique_edge_batch[inverse_indices, data.edge_batch] = 1
    assert torch.all(
        unique_edge_batch.sum(dim=-1) == 1
    )  # there should only be one batch id for each unique edge
    edge_batch = unique_edge_batch.nonzero().T[1]
    edge_batch = torch.cat((edge_batch, singleton_batch))
    rev_offset = edge_batch * data.num_nodes
    unique_edges = torch.cat((unique_edges, self_loops), dim=1)

    edge_index = unique_edges - rev_offset

    return edge_index, edge_batch, mask
