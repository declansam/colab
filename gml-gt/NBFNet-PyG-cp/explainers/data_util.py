import torch
import copy
from nbfnet import tasks
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


"""
This file contains utilities functions pertaining to the data structures used for explanations.
"""


def batched_k_hop_subgraph(
    node_idx, num_hops, edge_index, num_nodes, return_distance=False
):
    """
    gets the k_hop_subgraph in a batched fashion

    Args:
        node_idx: the central node indices (batch_size, num_nodes_per_triple)
        num_hops: the number of hops to consider
        edge_index: the graph edge_index
        num_nodes: the graph number of nodes
        return_distance: whether to return the min distance of each edge from the central node

    Returns:
        node_mask: (batch_size, num_nodes) indicating whether a node belongs in the k-hop neighbor of each central node
        edge_mask: (batch_size, num_edges) indicating whether an edge belongs in the k-hop neighbor of each central node
    """
    col, row = edge_index

    node_mask = row.new_empty((node_idx.size(0), num_nodes), dtype=torch.bool)
    # stores the edges that was included within the num hops
    edge_mask = row.new_empty((node_idx.size(0), row.size(0)), dtype=torch.bool).fill_(
        False
    )
    # stores whether the tgt node of each edge was included last hop
    tgt_edge_mask = row.new_empty((node_idx.size(0), row.size(0)), dtype=torch.bool)
    # stores whether the src node of each edge was included last hop
    src_edge_mask = row.new_empty((node_idx.size(0), row.size(0)), dtype=torch.bool)
    edge_distance = row.new_empty(
        (node_idx.size(0), row.size(0)), dtype=torch.float32
    ).fill_(float("inf"))

    batch = torch.arange(node_idx.size(0)).repeat_interleave(node_idx.size(1))
    node_idx = node_idx.to(row.device).flatten()

    preserved_node_mask = torch.zeros_like(node_mask)

    for k in range(num_hops):
        node_mask.fill_(False)
        node_mask[batch, node_idx] = True

        # check if the tgt node of each edge was included last hop
        torch.index_select(node_mask, 1, row, out=tgt_edge_mask)
        # check if the src node of each edge was included last hop
        torch.index_select(node_mask, 1, col, out=src_edge_mask)

        # get the current distance for the selected edges
        dist = torch.tensor([k + 1], device=tgt_edge_mask.device).repeat(
            tgt_edge_mask.sum()
        )
        # update the edge distance accordingly
        edge_distance[tgt_edge_mask] = torch.where(
            dist < edge_distance[tgt_edge_mask], dist, edge_distance[tgt_edge_mask]
        )
        dist = torch.tensor([k + 1], device=tgt_edge_mask.device).repeat(
            src_edge_mask.sum()
        )
        # update the edge distance accordingly
        edge_distance[src_edge_mask] = torch.where(
            dist < edge_distance[src_edge_mask], dist, edge_distance[src_edge_mask]
        )
        edge_mask |= tgt_edge_mask
        edge_mask |= src_edge_mask

        preserved_node_mask |= node_mask
        # get the row (batch) index and col (edge) index for the edges whose tgt node was included last hop
        batch, edge_id = tgt_edge_mask.nonzero().T
        # corresponds to the src nodes whose tgt node was included last hop
        node_idx = col[edge_id]

    # mark the src nodes from the final hop in the preserved node mask
    node_mask.fill_(False)
    node_mask[batch, node_idx] = True
    preserved_node_mask |= node_mask

    if return_distance:
        return preserved_node_mask, edge_mask, edge_distance

    return preserved_node_mask, edge_mask


def prepare_full_data(
    data, batch_size, central_nodes=None, hops=None, distance_dropout=False
):
    """
    Prepare the full data
    """
    if distance_dropout:
        # get the edge distance from the central nodes
        _, _, edge_distance = batched_k_hop_subgraph(
            central_nodes, hops, data.edge_index, data.num_nodes, return_distance=True
        )

    device = data.edge_index.device
    
    # Necessary copy to not modify the split data
    s_data = copy.copy(data)

    # store the original edge index and edge type
    s_data.original_edge_index = data.edge_index
    s_data.original_edge_type = data.edge_type
    
    # repeat the edge index and edge type for each batch
    '''
    s_data.edge_index = data.edge_index.repeat(1, batch_size)  # Repeat edges
    s_data.edge_type = data.edge_type.repeat(batch_size)       # Repeat edge types
    s_data.node_batch = torch.arange(batch_size).repeat_interleave(data.num_nodes)  # [0,0,...,1,1,...,2,2,...]
    '''
    s_data.edge_index = data.edge_index.repeat(1, batch_size)
    s_data.edge_type = data.edge_type.repeat(batch_size)

    # Build node-level bookkeeping
    # node ids repeat for each batch
    # Shape = [num_nodes * batch_size]
    s_data.node_id = torch.arange(data.num_nodes, device=device).repeat(batch_size)

    # For each node, record which batch copy it belongs to.
    s_data.node_batch = torch.arange(batch_size, device=device).repeat_interleave(
        data.num_nodes
    )

    # Build edge-level bookkeeping
    s_data.edge_batch = torch.arange(batch_size, device=device).repeat_interleave(
        data.edge_index.size(1)
    )

    # Number of nodes and edges per batch copy
    # For each copy, store the number of nodes/edges (same as original).
    s_data.subgraph_num_nodes = torch.tensor([data.num_nodes], device=device).repeat(
        batch_size
    )
    s_data.subgraph_num_edges = torch.tensor(
        [data.edge_index.size(1)], device=device
    ).repeat(batch_size)

    # For each copy, store the max number of nodes/edges across all queries in that copy.
    s_data.max_num_nodes = data.num_nodes
    s_data.central_node_index = central_nodes

    if distance_dropout:
        s_data.edge_distance = edge_distance.flatten()

    return s_data


def prepare_subgraph_data(
    data, central_nodes, node_mask, edge_mask, count_components=False
):
    """
    Given the node_mask and edge_mask, this creates a subgraph data for each query along with its central nodes.
    """
    # create relabeled edge_index and x
    node_batch, node_id = node_mask.nonzero().T
    edge_batch, edge_id = edge_mask.nonzero().T
    num_nodes = node_mask.sum(dim=1)
    num_edges = edge_mask.sum(dim=1)
    max_num_nodes = max(num_nodes)
    max_num_edges = max(num_edges)

    # nodes
    # make an offset so that each node_id will be unique
    offsets = node_batch * data.num_nodes
    # the unique_node_id for all of the nodes (sorted)
    unique_node_id = node_id + offsets
    # edges
    # make an offset so that each node in the edge will match the unique_node_id
    offsets = (edge_batch * data.num_nodes).unsqueeze(0)
    unique_edges = data.edge_index[:, edge_id] + offsets
    # relabel the node_id in edge_index
    assert torch.all(torch.isin(unique_edges, unique_node_id))
    # assign for every node in edges what indicies of unique_node_id it is
    edge_index = torch.bucketize(unique_edges, unique_node_id, right=False)

    if count_components:
        graph_data = Data(edge_index=edge_index, num_nodes=num_nodes.sum())
        G = to_networkx(graph_data)
        count = nx.number_weakly_connected_components(G)
        # count = 0
        # for c in C:
        #     # ignore singletons
        #     assert len(c) != 0
        #     count += 1

        return count / central_nodes.size(0)

    batch_offsets = torch.cumsum(num_nodes, dim=0)
    batch_offsets -= num_nodes  # indicates the offsets to relabel the indices
    offsets = batch_offsets[edge_batch].unsqueeze(0)
    edge_index -= offsets
    # map the central nodes
    offsets = (
        torch.arange(central_nodes.size(0), device=central_nodes.device)
        * data.num_nodes
    ).unsqueeze(1)
    unique_central_node_id = central_nodes + offsets
    assert torch.all(torch.isin(unique_central_node_id, unique_node_id))
    central_node_index = torch.bucketize(
        unique_central_node_id, unique_node_id, right=False
    )
    central_node_index -= batch_offsets.unsqueeze(1)
    # Create subgraph data
    s_data = copy.copy(data)
    s_data.original_edge_index = data.edge_index
    s_data.original_edge_type = data.edge_type
    s_data.edge_index = edge_index
    s_data.edge_type = data.edge_type[edge_id]
    s_data.node_id = node_id
    s_data.node_batch = node_batch
    s_data.edge_batch = edge_batch
    s_data.subgraph_num_nodes = num_nodes
    s_data.subgraph_num_edges = num_edges
    s_data.max_num_nodes = max_num_nodes
    s_data.central_node_index = central_node_index
    return s_data


def prepare_dist_data(
    data,
    central_nodes,
    hops,
    eval_on_edge_drop=0,
    randomized_edge_drop=0,
    eval_dropout_distance=0,
    max_edge_drop_prob=0.9,
):
    # if input does not specify dropout, return the unmasked edges
    if not (
        (data.split == "train" or eval_on_edge_drop)
        and (randomized_edge_drop > 0 or eval_dropout_distance > 0)
    ):
        node_mask = torch.ones(
            (central_nodes.size(0), data.num_nodes),
            dtype=torch.bool,
            device=central_nodes.device,
        )
        edge_mask = torch.ones(
            (central_nodes.size(0), data.edge_index.size(1)),
            dtype=torch.bool,
            device=central_nodes.device,
        )

    else:
        # get the edge distance from the central nodes
        _, _, edge_distance = batched_k_hop_subgraph(
            central_nodes, hops, data.edge_index, data.num_nodes, return_distance=True
        )
        # for each subgraph, mask the edge according to the probability
        if randomized_edge_drop > 0:
            # you cannot have both the eval dropout distance and edge drop probability
            assert eval_dropout_distance <= 0
            edge_drop_prob = (randomized_edge_drop * edge_distance).clamp(
                max=max_edge_drop_prob
            )
            edge_mask = (
                torch.rand(edge_drop_prob.shape, device=edge_drop_prob.device)
                > edge_drop_prob
            )
        # hard dropping of edges based on distance
        elif eval_dropout_distance > 0:
            edge_mask = edge_distance <= eval_dropout_distance

        # from the edge mask, determine which nodes were included
        batch_id, edge_id = edge_mask.nonzero().T
        src_node_id = data.edge_index[0][edge_id]
        tgt_node_id = data.edge_index[1][edge_id]
        node_mask = torch.zeros(
            (central_nodes.size(0), data.num_nodes),
            dtype=torch.bool,
            device=edge_mask.device,
        )
        node_mask[batch_id, src_node_id] = True
        node_mask[batch_id, tgt_node_id] = True
        # ensure that the central node (i.e. the head of the query) is included in each subgraph
        node_mask[torch.arange(central_nodes.size(0)), central_nodes] = True

    return prepare_subgraph_data(data, central_nodes, node_mask, edge_mask)


def prepare_ego_data(data, central_nodes, hops):
    """
    Prepare the k_hop_graph and create subgraph data.
    This will prepare the edge index s.t. it only consist of indicies from 0, ..., max_num_nodes - 1,
    where max_num_nodes is the max num nodes across all the queries in a batch.
    We do this by first assigning unique_node_id to the nodes in the k_hop_graph of the central_nodes of the batch.
    And then, the edge_index of the k_hop_graph will also use the unique_node_id.
    We bucketize the node id in this edge_index, s.t. each node id becomes there index in unique_node_id
    We relabel the nodes by subtracting the offsets (the cumsum of the number of nodes before that instance in the batch)
    Args:
        data: the full split data
        central_nodes: the central nodes for each query which you make the ego network, shape = (batch_size, num_central_nodes)
        hops: the number of hops for the ego network
    Returns:
        s_data: the subgraph data
    """
    # get the k_hop_subgraph
    k_hop_nodes, k_hop_edges = batched_k_hop_subgraph(
        central_nodes, hops, data.edge_index, data.num_nodes
    )
    return prepare_subgraph_data(data, central_nodes, k_hop_nodes, k_hop_edges)


def prepare_expl_data(
    cfg,
    data,
    central_nodes,
    batch,
    train_on_counter_factual=False,
    count_components=False,
    ignore_center=False,
):
    h_index, t_index, r_index = batch.unbind(-1)  # (batch_size, num_triples)
    # assumption: the batch has been converted to tail batch
    assert (h_index[:, [0]] == h_index).all()
    assert (r_index[:, [0]] == r_index).all()
    # prepare the query (h, r)
    query = batch[:, 0, :]
    query = query[:, [0, 2]].to(torch.int32)
    # find the matching explanations
    all_query = data.expl[0]
    row_id, num_match = tasks.edge_match(all_query.T, query.T)
    assert torch.all(num_match == 1)
    expl = data.expl[1][row_id]
    # filter out the padding tokens (-1s)

    # the explanations are indicated by the nodes
    if hasattr(cfg.explainer_eval, "node_mask") and cfg.explainer_eval.node_mask:
        num_nodes = (expl >= 0).sum(1)
        batch_id = torch.arange(query.size(0), device=query.device).repeat_interleave(
            num_nodes
        )
        node_id = expl[expl >= 0]
        unique_node_id = node_id + batch_id * data.num_nodes
        edge_batch = (
            torch.arange(query.size(0), device=query.device)
            .repeat_interleave(data.edge_index.size(1))
            .unsqueeze(0)
        )
        unique_edge_index = (
            data.edge_index.repeat(1, query.size(0)) + edge_batch * data.num_nodes
        )
        # get the edges of the induced subgraph
        edge_mask = torch.all(torch.isin(unique_edge_index, unique_node_id), dim=0)
        edge_mask = edge_mask.reshape(query.size(0), data.edge_index.size(1))
        if train_on_counter_factual:
            # include all the nodes
            node_mask = torch.ones(
                (query.size(0), data.num_nodes), dtype=torch.bool, device=query.device
            )
            # exclude edges from the induced graphs
            edge_mask = ~edge_mask
        else:
            node_mask = torch.zeros(
                (query.size(0), data.num_nodes), dtype=torch.bool, device=query.device
            )
            node_mask[batch_id, node_id] = True
            # ensure that the central node (i.e. the head of the query) is included in each subgraph
            node_mask[torch.arange(central_nodes.size(0)), central_nodes] = True

    else:
        num_edges = (expl >= 0).sum(1)
        batch_id = torch.arange(query.size(0), device=query.device).repeat_interleave(
            num_edges
        )
        edge_id = expl[expl >= 0]
        # from the edge mask, determine which nodes were included
        src_node_id = data.edge_index[0][edge_id]
        tgt_node_id = data.edge_index[1][edge_id]
        # create the node and edge mask for each query
        # if we are training on counter factual explanations
        if train_on_counter_factual:
            # include all the nodes
            node_mask = torch.ones(
                (query.size(0), data.num_nodes), dtype=torch.bool, device=query.device
            )
            edge_mask = torch.ones(
                (query.size(0), data.edge_index.size(1)),
                dtype=torch.bool,
                device=query.device,
            )
            # do not include the edges selected by the explainer
            edge_mask[batch_id, edge_id] = False
        # if we are training on factual explanations
        else:
            node_mask = torch.zeros(
                (query.size(0), data.num_nodes), dtype=torch.bool, device=query.device
            )
            edge_mask = torch.zeros(
                (query.size(0), data.edge_index.size(1)),
                dtype=torch.bool,
                device=query.device,
            )
            edge_mask[batch_id, edge_id] = True
            node_mask[batch_id, src_node_id] = True
            node_mask[batch_id, tgt_node_id] = True
            if not ignore_center:
                # ensure that the central node (i.e. the head of the query) is included in each subgraph
                node_mask[torch.arange(central_nodes.size(0)), central_nodes] = True

    return prepare_subgraph_data(
        data, central_nodes, node_mask, edge_mask, count_components=count_components
    )


def prepare_ego_data_rgcn(data, central_nodes, hops):
    """Prepares the ego data for RGCN"""
    # get the k_hop_subgraph
    k_hop_nodes, k_hop_edges = batched_k_hop_subgraph(
        central_nodes, hops, data.edge_index, data.num_nodes
    )
    # create relabeled edge_index and x
    node_batch, node_id = k_hop_nodes.nonzero().T
    edge_batch, edge_id = k_hop_edges.nonzero().T
    num_nodes = k_hop_nodes.sum(dim=1)
    num_edges = k_hop_edges.sum(dim=1)
    max_num_nodes = max(num_nodes)
    max_num_edges = max(num_edges)

    # nodes
    # make an offset so that each node_id will be unique
    offsets = node_batch * data.num_nodes
    # the unique_node_id for all of the nodes (sorted)
    unique_node_id = node_id + offsets

    # edges
    # make an offset so that each node in the edge will match the unique_node_id
    offsets = (edge_batch * data.num_nodes).unsqueeze(0)
    unique_edges = data.edge_index[:, edge_id] + offsets

    # relabel the node_id in edge_index
    assert torch.all(torch.isin(unique_edges, unique_node_id))
    # assign for every node in edges what indicies of unique_node_id it is
    edge_index = torch.bucketize(unique_edges, unique_node_id, right=False)

    # map the central nodes
    offsets = (
        torch.arange(central_nodes.size(0), device=central_nodes.device)
        * data.num_nodes
    ).unsqueeze(1)
    unique_central_node_id = central_nodes + offsets
    assert torch.all(torch.isin(unique_central_node_id, unique_node_id))
    central_node_index = torch.bucketize(
        unique_central_node_id, unique_node_id, right=False
    )

    s_data = copy.copy(data)
    s_data.original_edge_index = data.edge_index
    s_data.original_edge_type = data.edge_type
    s_data.edge_index = edge_index
    s_data.edge_type = data.edge_type[edge_id]
    s_data.node_id = node_id
    s_data.node_batch = node_batch
    s_data.edge_batch = edge_batch
    s_data.subgraph_num_nodes = num_nodes
    s_data.subgraph_num_edges = num_edges
    s_data.max_num_nodes = max_num_nodes
    s_data.central_node_index = central_node_index
    s_data.total_num_nodes = num_nodes.sum()
    s_data.x = data.x[node_id]
    s_data.ntypes = data.ntypes[node_id]

    return s_data


def map_groundtruth(data, groundtruth_data, src_tgt, label):
    """
    Map the ground truth data to the respective node ids
    """
    if groundtruth_data is None:
        return data
    pos_edge_mask = label == 1
    pos_edges = src_tgt[pos_edge_mask]
    # map the pos edges to the ID
    ids = []
    for pos_edge in pos_edges:
        id = groundtruth_data.posEdge2id[(tuple(pos_edge.tolist()))]
        ids.append(id)
    ids = torch.tensor(ids, device=label.device)  # the id for each pos edge (Unsorted!)

    # get the corresponding ground truth edges
    edge_mask = torch.isin(groundtruth_data.gt_id, ids)
    gt_edge_index = groundtruth_data.gt_edge_index[:, edge_mask]
    gt_edge_type = groundtruth_data.gt_edge_type[edge_mask]
    gt_id = groundtruth_data.gt_id[edge_mask]  # the id for each pos edge (Sorted!)
    # we need to get the batch id for each ground truth edge
    # get the batch id for each pos edge
    batch_id = torch.arange(len(label), device=label.device)[pos_edge_mask]
    # ! The ground truth edges are sorted based on the ids
    # by permuting the batch_id with the indices used for sorting the ids,
    # we can get match the batch_id to each ground truth edge
    _, indices = torch.sort(ids)
    # permute it s.t. the batch id matches the Sorted gt_id!
    batch_id = batch_id[indices]
    # get the number of edges for each gt_id
    _, count = torch.unique(gt_id, return_counts=True)
    gt_edge_batch = batch_id.repeat_interleave(count)

    # now that we have the batch_id, relabel the indices to match each node
    offsets = data.node_batch * data.num_nodes
    unique_node_id = data.node_id + offsets
    offsets = (gt_edge_batch * data.num_nodes).unsqueeze(0)
    unique_edges = gt_edge_index + offsets

    # relabel the node_id in edge_index
    assert torch.all(torch.isin(unique_edges, unique_node_id))
    gt_edge_index = torch.bucketize(unique_edges, unique_node_id, right=False)
    batch_offsets = torch.cumsum(data.subgraph_num_nodes, dim=0)
    batch_offsets -= data.subgraph_num_nodes
    offsets = batch_offsets[gt_edge_batch].unsqueeze(0)
    gt_edge_index -= offsets

    data.gt_edge_index = gt_edge_index
    data.gt_edge_type = gt_edge_type
    data.gt_edge_batch = gt_edge_batch

    return data


def get_groundtruth(data, groundtruth_data, src_tgt):
    """
    Map the ground truth data to the respective node ids
    """
    if groundtruth_data is None:
        return data
    pos_edges = src_tgt
    # map the pos edges to the ID
    ids = []
    for pos_edge in pos_edges:
        id = groundtruth_data.posEdge2id[(tuple(pos_edge.tolist()))]
        ids.append(id)
    # the id for each pos edge (Unsorted!)
    ids = torch.tensor(ids, device=pos_edges.device)

    # get the corresponding ground truth edges
    edge_mask = torch.isin(groundtruth_data.gt_id, ids)
    gt_edge_index = groundtruth_data.gt_edge_index[:, edge_mask]
    gt_edge_type = groundtruth_data.gt_edge_type[edge_mask]
    gt_id = groundtruth_data.gt_id[edge_mask]  # the id for each pos edge (Sorted!)
    # we need to get the batch id for each ground truth edge
    # get the batch id for each pos edge
    batch_id = torch.arange(len(pos_edges), device=pos_edges.device)
    # ! The ground truth edges are sorted based on the ids
    # by permuting the batch_id with the indices used for sorting the ids,
    # we can get match the batch_id to each ground truth edge
    _, indices = torch.sort(ids)
    # permute it s.t. the batch id matches the Sorted gt_id!
    batch_id = batch_id[indices]
    # get the number of edges for each gt_id
    _, count = torch.unique(gt_id, return_counts=True)
    gt_edge_batch = batch_id.repeat_interleave(count)

    data.gt_edge_index = gt_edge_index
    data.gt_edge_type = gt_edge_type
    data.gt_edge_batch = gt_edge_batch

    return data


def check_if_tail_in_network(ego_data, batch):
    """
    In the test set case, we will first have to check whether the tails are inside the network
    In training, this is already done as we only sample the negatives within the network
        ego_data: the batched ego network data
        batch: the batch, assumed to have been converted to tail batch
    """
    batch_size = len(batch)
    ego_mask = torch.zeros(
        (batch_size, ego_data.num_nodes), device=batch.device, dtype=torch.bool
    )
    ego_mask[ego_data.node_batch, ego_data.node_id] = True
    batch_id = torch.arange(batch_size).repeat_interleave(ego_data.num_nodes)
    tail_idx = torch.arange(ego_data.num_nodes).repeat(batch_size)
    tail_inside = ego_mask[batch_id, tail_idx].view(batch_size, -1)
    tails = torch.ones_like(batch[:, :, 1]) * -1
    tails[tail_inside] = batch[:, :, 1][tail_inside]
    batch[:, :, 1] = tails
    assert torch.equal((batch[:, :, 1] >= 0).sum(dim=-1), ego_data.subgraph_num_nodes)
    return batch


def relabel_batch(data, batch):
    """
    Relabel the indices in batch
    The h_index and t_index should be relabeled based on the new node_id
    However, there is no guarantee that the tail nodes are inside the ego_network.
    Therefore, these will take an index of -1
    Important! The relabeling of the nodes has to match how edge_index was relabeled (prepare_ego_data)!
    Args:
        data: the egonetwork subgraph data
        batch: the batch of the triples
    Returns:
        (mapped_batch, valid_tails) where valid_tails are a mask that has True for tail nodes inside the ego network.
    """
    h_index, t_index, r_index = batch.unbind(-1)
    B = h_index.size(0)  # batch size

    # nodes
    offsets = data.node_batch * data.num_nodes
    unique_node_id = data.node_id + offsets

    batch_offsets = torch.cumsum(data.subgraph_num_nodes, dim=0)
    batch_offsets -= data.subgraph_num_nodes

    # Relabel the batch
    batch_index = (
        torch.arange(B, device=h_index.device).unsqueeze(1).expand(h_index.shape)
    )
    reverse_offsets = batch_offsets[batch_index]
    offsets = batch_index * data.num_nodes

    # the head index (which will always be included)
    unique_h_index = h_index + offsets
    # all the heads must be inside the ego network
    assert torch.all(torch.isin(unique_h_index, unique_node_id))
    h_index = torch.bucketize(unique_h_index, unique_node_id, right=False)
    h_index -= reverse_offsets

    mappable_tails = t_index != -1
    unique_t_index = t_index[mappable_tails] + offsets[mappable_tails]
    # all the tails (which is inside the ego network) has to be mappable
    assert torch.all(torch.isin(unique_t_index, unique_node_id))
    mapped_t_index = torch.bucketize(unique_t_index, unique_node_id, right=False)
    mapped_t_index -= reverse_offsets[mappable_tails]
    t_index[mappable_tails] = mapped_t_index
    assert torch.max(h_index) < torch.max(data.subgraph_num_nodes) and torch.max(
        t_index
    ) < torch.max(data.subgraph_num_nodes)

    return torch.stack([h_index, t_index, r_index], dim=-1), mappable_tails


def create_batched_data(data):
    """
    Creates the batched data such that NBFNet can process different graphs at the same time.
    Args:
        data: the subgraph data
    Returns:
        batched_data
    """
    # Assumption: the edges are sorted based on the batches
    prev = data.edge_batch[:-1]  # get the prev edge batch
    # get the diff of the prev edge batch - the next edge batch
    check = prev - data.edge_batch[1:]
    assert not torch.any(check > 0)  # if it is sorted, the diff should be <= 0
    # prepare the batched edge index
    batch_size = len(data.subgraph_num_edges)
    max_num_edges = max(data.subgraph_num_edges)
    batched_edge_index = torch.zeros(
        (2, batch_size, max_num_edges),
        dtype=data.edge_index.dtype,
        device=data.edge_index.device,
    )
    edge_filter = torch.arange(max_num_edges, device=batched_edge_index.device)
    edge_filter = edge_filter.view(1, 1, -1).expand_as(batched_edge_index)
    edge_filter = edge_filter < data.subgraph_num_edges.view(1, -1, 1)
    batched_edge_index[edge_filter] = data.edge_index.flatten()
    edge_filter = edge_filter[0]
    batched_edge_type = torch.zeros(
        (batch_size, max_num_edges),
        dtype=data.edge_type.dtype,
        device=data.edge_type.device,
    )
    batched_edge_type[edge_filter] = data.edge_type
    # Add the batched edge info to the data
    data.batched_edge_index = batched_edge_index
    data.edge_filter = edge_filter.to(torch.float32)
    data.batched_edge_type = batched_edge_type
    return data


def remove_edges(data, remove_edge_mask):
    """removes edges from the batched data
    remove_edge_mask is True for edges in data.edge_index that will be removed
    """
    data = copy.copy(data)
    edge_mask = ~remove_edge_mask
    data.edge_index = data.edge_index[:, edge_mask]
    data.edge_type = data.edge_type[edge_mask]
    data.edge_batch = data.edge_batch[edge_mask]
    # ensure if no edge the num_edge will be 0
    index, counts = torch.unique(data.edge_batch, return_counts=True)
    subgraph_num_edges = torch.zeros_like(data.subgraph_num_edges)
    subgraph_num_edges[index] = counts
    data.subgraph_num_edges = subgraph_num_edges
    data = create_batched_data(data)
    return data


def recreate_data_object(data, edges, edge_type, edge_batch, edge_distance=None):
    # recreate the data object with only masked edges so that the masked inference is efficient
    data.edge_index = edges
    data.edge_type = edge_type
    data.edge_batch = edge_batch
    if edge_distance is not None:
        data.edge_distance = edge_distance
    # ensure if no edge the num_edge will be 0
    index, counts = torch.unique(data.edge_batch, return_counts=True)
    subgraph_num_edges = torch.zeros_like(data.subgraph_num_edges)
    subgraph_num_edges[index] = counts
    data.subgraph_num_edges = subgraph_num_edges
    data = create_batched_data(data)
    return data


def counter_factual_edge_filter(data, edge_mask=None, batch_id=None, edge_id=None):
    """
    This function modifies data.edge_filter such that the counter factual edges are dropped (gets value 0),
    and creates data.counter_factual_edge_filter which has value 1 for counter factual edges.
    There are two ways to specify the counter factual edges:
        1. Pass edge_mask which specifies for each edge in data.edge_index
            whether they are counter factual edge or not.
        2. Pass batch_id and edge_id, which corresponds to the index of the counter factual edges
            on data.batched_edge_index
    Args:
        data: the batched data
        edge_mask
        batch_id:
    Returns:
        data: the updated batched data
    """
    edge_filter = torch.zeros_like(data.edge_filter)
    if edge_mask is not None:
        edge_filter[data.edge_filter.to(torch.bool)] = 1 - edge_mask.to(
            data.edge_filter.dtype
        )
        cf_edge_filter = torch.zeros_like(data.edge_filter)
        cf_edge_filter[data.edge_filter.to(torch.bool)] = edge_mask.to(
            data.edge_filter.dtype
        )
        data.edge_filter = edge_filter
        data.counter_factual_filter = cf_edge_filter
    else:
        data.edge_filter[batch_id, edge_id] = 0  # drop the counter factual edges
        edge_filter[batch_id, edge_id] = 1
        data.counter_factual_filter = edge_filter
    num_edges = (data.edge_filter + data.counter_factual_filter).sum(dim=1)
    assert torch.equal(num_edges, data.subgraph_num_edges)
    return data


def create_s_node_embeds(data, node_embeds):
    batch_size = node_embeds.size(0)
    # get the relevant node embeds
    s_node_embeds = torch.zeros(
        (batch_size, data.max_num_nodes, node_embeds.size(-1)),
        device=node_embeds.device,
        dtype=node_embeds.dtype,
    )
    indices = torch.arange(data.max_num_nodes, device=node_embeds.device).repeat(
        batch_size, 1
    )
    s_filter = indices < data.subgraph_num_nodes.unsqueeze(1)
    s_node_embeds[s_filter] = node_embeds[data.node_batch, data.node_id]
    return s_node_embeds


def convert_multigraph_to_digraph(
    data, mask, adj_aggr, handle_singletons=False, return_edge_type=False
):
    """
    1. Knowledge graph can be multigraph, convert this to digraph (only one edge in u, v) by aggregating the weights for each relation.
    2. (handle_singletons): The central nodes / nodes in masked edges can be a singleton. Add self loop so the power method works.
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
    if adj_aggr == "max":
        multi_edge_weights = torch.zeros_like(multi_edge_weights).fill_(float("-inf"))
        multi_edge_weights[inverse_indices, data.edge_type] = mask
        # for each unique edge, take the relation with highest weight (lowest negative weight)
        mask, edge_type = torch.max(multi_edge_weights, dim=-1)
    elif adj_aggr == "mean":
        dense_edge_filter = torch.zeros_like(multi_edge_weights)
        dense_edge_filter[inverse_indices, data.edge_type] = 1
        norm = dense_edge_filter.sum(dim=-1)
        mask = torch.sum(multi_edge_weights, dim=-1) / norm
    elif adj_aggr == "sum":
        mask = torch.sum(multi_edge_weights, dim=-1)
    else:
        raise ValueError(f"Unknown adj_aggr type: {adj_aggr}")

    batch_size = data.edge_filter.size(0)
    # check if there are any singleton nodes
    # if so, add self loops so it doesn't mess up the random walk
    if handle_singletons:
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
    if handle_singletons:
        edge_batch = torch.cat((edge_batch, singleton_batch))
    rev_offset = edge_batch * data.num_nodes
    if handle_singletons:
        unique_edges = torch.cat((unique_edges, self_loops), dim=1)
    edge_index = unique_edges - rev_offset

    if adj_aggr == "max" and return_edge_type:
        return edge_index, edge_batch, mask, edge_type

    return edge_index, edge_batch, mask
