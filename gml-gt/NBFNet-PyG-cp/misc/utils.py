
import torch

def batched_k_hop_subgraph(node_idx, num_hops, edge_index, num_nodes):
    '''
    gets the k_hop_subgraph in a batched fashion

    Args:
        node_idx: the central node indices (batch_size,)
        num_hops: the number of hops to consider
        edge_index: the graph edge_index
        num_nodes: the graph number of nodes

    Returns:
        node_mask: (batch_size, num_nodes) indicating whether a node belongs in the k-hop neighbor of each central node
        edge_mask: (batch_size, num_edges) indicating whether an edge belongs in the k-hop neighbor of each central node
    '''
    col, row = edge_index

    node_mask = row.new_empty((node_idx.size(0), num_nodes), dtype=torch.bool)
    edge_mask = row.new_empty((node_idx.size(0), row.size(0)), dtype=torch.bool)

    node_idx = node_idx.to(row.device)
    batch = torch.arange(node_idx.size(0))

    preserved_edge_mask = torch.zeros_like(edge_mask)
    preserved_node_mask = torch.zeros_like(node_mask)

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[batch, node_idx] = True
        torch.index_select(node_mask, 1, row, out=edge_mask) # get if the tgt node of each edge was included last hop
        preserved_edge_mask |= edge_mask
        preserved_node_mask |= node_mask

        batch, edge_id = edge_mask.nonzero().T # get the row (batch) index and col (edge) index
        node_idx = col[edge_id] # corresponds to the src nodes whose tgt node was included last hop 
    # mark the src nodes from the final hop in the preserved node mask
    node_mask.fill_(False)
    node_mask[batch, node_idx] = True
    preserved_node_mask |= node_mask

    return preserved_node_mask, preserved_edge_mask
