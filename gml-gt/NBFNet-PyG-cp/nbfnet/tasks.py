from functools import reduce
from torch_geometric.data import Data
import torch


def edge_match(edge_index, query_index):
    # translate both indices by 1, in case any indices are -1 (padding value)
    edge_index += 1
    query_index += 1

    # O((n + q)logn) time
    # O(n) memory
    # edge_index: big underlying graph
    # query_index: edges to match

    # preparing unique hashing of edges, base: (max_node, max_relation) + 1
    query_base = query_index.max(dim=1)[0] + 1
    base = edge_index.max(dim=1)[0] + 1
    base = torch.max(query_base, base)
    # we will map edges to long ints, so we need to make sure the maximum product is less than MAX_LONG_INT
    # idea: max number of edges = num_nodes * num_relations
    # e.g. for a graph of 10 nodes / 5 relations, edge IDs 0...9 mean all possible outgoing edge types from node 0
    # given a tuple (h, r), we will search for all other existing edges starting from head h
    assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max
    scale = base.cumprod(0)
    scale = scale[-1] // scale

    # hash both the original edge index and the query index to unique integers
    edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)
    edge_hash, order = edge_hash.sort()
    query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)

    """
    Assume an edge is only a pair of (h, r)
    Assume node_id is 0 - 2, so we have 3 nodes, and rel_id is 0 - 1, so we have 2 rels.
    base = [3, 2]
    product = 3 * 2 = 6
    scale[0] (for the node) is 6/3 = 2. And scale[1] (for the rel) is 6/6 = 1.
    So, for an edge (0, 1), the hashed value will be 0 * 2 + 1 * 1 = 1.
    For an edge (1, 0), the hashed value will be 1 * 2 + 0 * 1 = 2. 
    h(0, 0) = 0, h(0, 1) = 1, h(1, 0) = 2, h(1, 1) = 3, h(2, 0) = 4, h(2, 1) = 5
    """

    # matched ranges: [start[i], end[i])
    start = torch.bucketize(query_hash, edge_hash)
    end = torch.bucketize(query_hash, edge_hash, right=True)
    # num_match shows how many edges satisfy the (h, r) pattern for each query in the batch
    num_match = end - start

    """
    The above finds for each query edge, how many matching edges we have.
    torch.bucketize() will try to bucketize the query_hash into edge_hash, by using the default right = False, 
    it will try to place the query hash after the first instance. (start)
    By setting right = True, it will place the query hash after the last instance (last)

    By comparing the difference in the indices of start and end for each query, you get how many matching edges you have per query.
    """

    # generate the corresponding ranges
    offset = num_match.cumsum(0) - num_match
    range = torch.arange(num_match.sum(), device=edge_index.device)
    range = range + (start - offset).repeat_interleave(num_match)

    # translate back both indices by -1 to get original value
    edge_index -= 1
    query_index -= 1

    return order[range], num_match


def negative_sampling(
    data, batch, num_negative, strict=True, is_synthetic=False, mode=None
):
    """
    Does Negative Sampling from full graph data
    Args:
        data: the full split graph data
        batch: the positive triples
        num_negative: the number of negative triples to sample
        strict: whether to do strict negative sampling using the split graph data
        is_synthetic: whether the dataset is synthetic
        mode: the mode of evaluation for each query
    """
    batch_size = len(batch)
    pos_h_index, pos_t_index, pos_r_index = batch.t()

    if mode is None:
        mode = torch.zeros((batch_size,), dtype=torch.bool, device=batch.device)
        mode[: batch_size // 2] = (
            1  # first half will be tail batch, and the second half will be head batch
        )
    if is_synthetic:
        data = Data(
            edge_index=data.target_edge_index,
            edge_type=data.target_edge_type,
            num_nodes=data.num_nodes,
        )

    # strict negative sampling vs random negative sampling
    if strict:
        t_mask, h_mask = strict_negative_mask(data, batch)
        t_mask = t_mask[mode]
        neg_t_candidate = t_mask.nonzero()[:, 1]
        num_t_candidate = t_mask.sum(dim=-1)
        # draw samples for negative tails
        rand = torch.rand(len(t_mask), num_negative, device=batch.device)
        index = (rand * num_t_candidate.unsqueeze(-1)).long()
        index = index + (num_t_candidate.cumsum(0) - num_t_candidate).unsqueeze(-1)
        neg_t_index = neg_t_candidate[index]

        h_mask = h_mask[~mode]
        neg_h_candidate = h_mask.nonzero()[:, 1]
        num_h_candidate = h_mask.sum(dim=-1)
        # draw samples for negative heads
        rand = torch.rand(len(h_mask), num_negative, device=batch.device)
        index = (rand * num_h_candidate.unsqueeze(-1)).long()
        index = index + (num_h_candidate.cumsum(0) - num_h_candidate).unsqueeze(-1)
        neg_h_index = neg_h_candidate[index]
    else:
        neg_index = torch.randint(
            data.num_nodes, (batch_size, num_negative), device=batch.device
        )
        neg_t_index, neg_h_index = neg_index[mode], neg_index[~mode]

    h_index = pos_h_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index = pos_t_index.unsqueeze(-1).repeat(1, num_negative + 1)
    r_index = pos_r_index.unsqueeze(-1).repeat(1, num_negative + 1)

    t_index[mode, 1:] = neg_t_index
    h_index[~mode, 1:] = neg_h_index

    return torch.stack([h_index, t_index, r_index], dim=-1)


def get_neg_index(t_mask, h_mask, num_negative):
    num_t_candidate = t_mask.sum(dim=-1)  # the number of t candidates for each query

    neg_t_index = torch.ones(
        (len(t_mask), num_negative), dtype=torch.long, device=t_mask.device
    )
    # If the ego-network doesn't contain any negatives (most likely the head is a singleton),
    # It will have -1 as the index of the tail
    no_negatives = num_t_candidate == 0
    neg_t_index[no_negatives] = -1

    # draw samples for negative tails
    neg_t_candidate = t_mask[~no_negatives].nonzero()[
        :, 1
    ]  # the nodes that are actual neg t candidate
    num_t_candidate = num_t_candidate[~no_negatives]
    rand = torch.rand(len(num_t_candidate), num_negative, device=t_mask.device)
    index = (
        rand * num_t_candidate.unsqueeze(-1)
    ).long()  # get a random num_t_candidate
    index = index + (num_t_candidate.cumsum(0) - num_t_candidate).unsqueeze(
        -1
    )  # select the index of the neg t by adjusting the offsets
    neg_t_index[~no_negatives] = neg_t_candidate[index]

    # ensure we are only sampling from the ego_network
    num_h_candidate = h_mask.sum(dim=-1)

    neg_h_index = torch.ones(
        (len(h_mask), num_negative), dtype=torch.long, device=h_mask.device
    )
    # If the ego-network doesn't contain any negatives (most likely the head is a singleton),
    # It will have -1 as the index of the tail
    no_negatives = num_h_candidate == 0
    neg_h_index[no_negatives] = -1

    # draw samples for negative heads
    neg_h_candidate = h_mask[~no_negatives].nonzero()[:, 1]
    num_h_candidate = num_h_candidate[~no_negatives]
    rand = torch.rand(len(num_h_candidate), num_negative, device=h_mask.device)
    index = (rand * num_h_candidate.unsqueeze(-1)).long()
    index = index + (num_h_candidate.cumsum(0) - num_h_candidate).unsqueeze(-1)
    neg_h_index[~no_negatives] = neg_h_candidate[index]

    return neg_t_index, neg_h_index


def negative_sampling_from_ego(
    data, ego_data, batch, num_negative, mode, strict=True, is_synthetic=False
):
    """
    Samples negative tails from the ego network subgraph.
    Args:
        data: the full split data
        ego_data: the ego network subgraph data
        batch: the positive triples
        mode: the mode of evaluation
        strict: whether to perform strict negative sampling using the split data
        is_synthetic: whether the dataset is synthetic
    """
    batch_size = len(batch)
    pos_h_index, pos_t_index, pos_r_index = batch.t()
    ego_mask = torch.zeros(
        (batch_size, data.num_nodes), device=batch.device, dtype=torch.bool
    )
    ego_mask[ego_data.node_batch, ego_data.node_id] = True
    # check if the positive node is included in the ego network
    pos_t_included = ego_mask[torch.arange(batch_size), pos_t_index]
    pos_h_included = ego_mask[torch.arange(batch_size), pos_h_index]
    if is_synthetic:
        data = Data(
            edge_index=data.target_edge_index,
            edge_type=data.target_edge_type,
            num_nodes=data.num_nodes,
        )
    # strict negative sampling vs random negative sampling
    if strict:
        t_mask, h_mask = strict_negative_mask(data, batch)
        # ensure we are only sampling from the ego_network
        t_mask = torch.logical_and(t_mask[mode], ego_mask[mode])
        h_mask = torch.logical_and(h_mask[~mode], ego_mask[~mode])
        # get num_negative + 1 negative samples in case the positive node is not included in the ego network
        neg_t_index, neg_h_index = get_neg_index(t_mask, h_mask, num_negative + 1)
    else:
        # get num_negative + 1 negative samples in case the positive node is not included in the ego network
        neg_t_index, neg_h_index = get_neg_index(
            ego_mask[mode], ego_mask[~mode], num_negative + 1
        )

    h_index = pos_h_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index = pos_t_index.unsqueeze(-1).repeat(1, num_negative + 1)
    r_index = pos_r_index.unsqueeze(-1).repeat(1, num_negative + 1)

    pos_t_included = pos_t_included[mode]
    pos_h_included = pos_h_included[~mode]

    # if the pos is included in the ego network, it will use that as the answer
    # if the pos is NOT included in the ego network, it will only have negative samples
    _t_index = t_index[mode]
    _t_index[:, 1:][pos_t_included] = neg_t_index[:, 1:][pos_t_included]
    _t_index[:, :][~pos_t_included] = neg_t_index[~pos_t_included]
    t_index[mode] = _t_index

    _h_index = h_index[~mode]
    _h_index[:, 1:][pos_h_included] = neg_h_index[:, 1:][pos_h_included]
    _h_index[:, :][~pos_h_included] = neg_h_index[~pos_h_included]
    h_index[~mode] = _h_index

    pos_included = torch.cat((pos_t_included, pos_h_included))

    return torch.stack([h_index, t_index, r_index], dim=-1), pos_included


def negative_sample_to_tail(h_index, t_index, r_index, mode, num_relation):
    # convert p(h | t, r) to p(t' | h', r')
    # h' = t, r' = r^{-1}, t' = h
    if mode is not None:
        is_t_neg = mode.unsqueeze(1).to(torch.bool)
    else:
        is_t_neg = (h_index == h_index[:, [0]]).all(
            dim=-1, keepdim=True
        )  # if the head index changes, it knows that it's head-batch, True indicates Tail Batch
    new_h_index = torch.where(is_t_neg, h_index, t_index)
    new_t_index = torch.where(is_t_neg, t_index, h_index)
    new_r_index = torch.where(is_t_neg, r_index, r_index + num_relation // 2)
    return new_h_index, new_t_index, new_r_index


def conversion_to_tail_prediction(batch, num_relation, mode=None):
    """Convert the batch to tail prediction"""
    h_index, t_index, r_index = batch.unbind(-1)  # (batch_size, num_triples)
    # turn all triples in a batch into a tail prediction mode
    h_index, t_index, r_index = negative_sample_to_tail(
        h_index, t_index, r_index, mode, num_relation
    )
    if mode is None:  # we have some padded index in the explainer case
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
    batch = torch.stack([h_index, t_index, r_index], dim=-1)
    return batch


def all_negative(data, batch):
    pos_h_index, pos_t_index, pos_r_index = batch.t()
    r_index = pos_r_index.unsqueeze(-1).expand(-1, data.num_nodes)
    # generate all negative tails for this batch
    all_index = torch.arange(data.num_nodes, device=batch.device)
    h_index, t_index = torch.meshgrid(pos_h_index, all_index, indexing="ij")
    t_batch = torch.stack([h_index, t_index, r_index], dim=-1)
    # generate all negative heads for this batch
    all_index = torch.arange(data.num_nodes, device=batch.device)
    t_index, h_index = torch.meshgrid(pos_t_index, all_index, indexing="ij")
    h_batch = torch.stack([h_index, t_index, r_index], dim=-1)
    return t_batch, h_batch


def strict_negative_mask(
    data, batch
):  # True -> Negative Sample, False -> Positive Sample, Target will have False
    # this function makes sure that for a given (h, r) batch we will NOT sample true tails as random negatives
    # similarly, for a given (t, r) we will NOT sample existing true heads as random negatives

    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # part I: sample hard negative tails
    # edge index of all (head, relation) edges from the underlying graph
    edge_index = torch.stack([data.edge_index[0], data.edge_type])
    # edge index of current batch (head, relation) for which we will sample negatives
    query_index = torch.stack([pos_h_index, pos_r_index])
    # search for all true tails for the given (h, r) batch
    edge_id, num_t_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    t_truth_index = data.edge_index[1, edge_id]
    sample_id = torch.arange(len(num_t_truth), device=batch.device).repeat_interleave(
        num_t_truth
    )
    t_mask = torch.ones(
        len(num_t_truth), data.num_nodes, dtype=torch.bool, device=batch.device
    )
    # assign 0s to the mask with the found true tails
    t_mask[sample_id, t_truth_index] = 0
    t_mask.scatter_(1, pos_t_index.unsqueeze(-1), 0)

    # part II: sample hard negative heads
    # edge_index[1] denotes tails, so the edge index becomes (t, r)
    edge_index = torch.stack([data.edge_index[1], data.edge_type])
    # edge index of current batch (tail, relation) for which we will sample heads
    query_index = torch.stack([pos_t_index, pos_r_index])
    # search for all true heads for the given (t, r) batch
    edge_id, num_h_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    h_truth_index = data.edge_index[0, edge_id]
    sample_id = torch.arange(len(num_h_truth), device=batch.device).repeat_interleave(
        num_h_truth
    )
    h_mask = torch.ones(
        len(num_h_truth), data.num_nodes, dtype=torch.bool, device=batch.device
    )
    # assign 0s to the mask with the found true heads
    h_mask[sample_id, h_truth_index] = 0
    h_mask.scatter_(1, pos_h_index.unsqueeze(-1), 0)

    return t_mask, h_mask


def compute_ranking(pred, target, mask=None):
    pos_pred = pred.gather(-1, target.unsqueeze(-1))
    if mask is not None:
        # filtered ranking
        ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
    else:
        # unfiltered ranking
        ranking = torch.sum(pos_pred <= pred, dim=-1) + 1
    return ranking


def compute_ranking_ego(pred, target, mask=None):
    pred[~valid_tails] = float("-inf")
    pos_pred = pred.gather(-1, target.unsqueeze(-1))
    if mask is not None:
        # filtered ranking
        ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
    else:
        # unfiltered ranking
        ranking = torch.sum(pos_pred <= pred, dim=-1) + 1
    return ranking


def get_topk(pred, topk, positives=None):
    """returns the topk predicted entities. No masking is done
    positives has True for positive entities for the given query except the one we are considering
    """
    if positives is not None:
        pred[positives] = float("-inf")
    argsort = torch.argsort(pred, dim=1, descending=True)
    if topk == "all":
        return argsort
    else:
        return argsort[:, :topk]
