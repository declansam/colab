def multigraph_to_graph(data, mask, neg_path_score):
    """
    Knowledge graph can be multigraph, convert this to digraph (only one edge in u, v) by selecting the relation with highest weight.
    (we select the relation with highest weight by selecting the relation of lowest neg weight!)
    This is necessary as Dijkstra's algorithm is not implemented for multigraph & we don't care about the relation with lower weight.
    We only select the relation with highest weight because the path finding algorithm will try to find the path with highest score.
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
    ).fill_(float("inf"))
    multi_edge_weights[inverse_indices, data.edge_type] = neg_path_score
    # for each unique edge, take the relation with highest weight (lowest negative weight)
    neg_path_score, edge_type = torch.min(multi_edge_weights, dim=-1)

    # for the relations with highest weight, get their corresponding edge mask
    multi_edge_mask = torch.zeros(
        (unique_edges.size(1), num_relations), device=unique_edges.device
    ).fill_(float("inf"))
    multi_edge_mask[inverse_indices, data.edge_type] = mask
    path_mask = multi_edge_mask[torch.arange(unique_edges.size(1)), edge_type]
    assert ~torch.any(path_mask == float("inf"))

    # we also need to keep the edge mask that were not selected!
    orig_edge_id = torch.zeros(
        (unique_edges.size(1), num_relations),
        device=unique_edges.device,
        dtype=torch.long,
    )
    # for each unique edges and their rels, get the original edge id
    orig_edge_id[inverse_indices, data.edge_type] = torch.arange(
        data.edge_type.size(0), device=unique_edges.device
    )
    # get the mask for the unselected edges
    unselected_edge_mask = torch.zeros_like(orig_edge_id, dtype=torch.bool)
    unselected_edge_mask[inverse_indices, data.edge_type] = True
    unselected_edge_mask[torch.arange(unique_edges.size(1)), edge_type] = (
        False  # mark the selected ones as false
    )
    unselected_edge_id = orig_edge_id[unselected_edge_mask]
    # the addition of selected and unselected should be equal to everything!
    assert path_mask.size(0) + unselected_edge_id.size(0) == mask.size(0)

    # map the unique edges back to its original
    # get the edge_batch for the unique edges
    batch_size = data.edge_filter.size(0)
    unique_edge_batch = torch.zeros(
        (unique_edges.size(1), batch_size), device=unique_edges.device, dtype=torch.long
    )
    unique_edge_batch[inverse_indices, data.edge_batch] = 1
    assert torch.all(
        unique_edge_batch.sum(dim=-1) == 1
    )  # there should only be one batch id for each unique edge
    edge_batch = unique_edge_batch.nonzero().T[1]
    rev_offset = edge_batch * data.num_nodes
    edge_index = unique_edges - rev_offset

    # digraph used specifically for path finding
    path_data = copy.copy(data)
    path_data.edge_index = edge_index
    path_data.edge_type = edge_type
    path_data.edge_batch = edge_batch
    index, counts = torch.unique(
        path_data.edge_batch, return_counts=True
    )  # ensure if no edge the num_edge will be 0
    subgraph_num_edges = torch.zeros_like(path_data.subgraph_num_edges)
    subgraph_num_edges[index] = counts
    path_data.subgraph_num_edges = subgraph_num_edges
    path_data = create_batched_data(path_data)

    return path_data, path_mask, neg_path_score, unselected_edge_id
