import torch
from nbfnet.tasks import edge_match


def negative_sampling(data, num_neg_samples):
    # get the source type
    src = data.target_edge_index[0]
    src_type = data.ntypes[src][0]
    assert torch.all(data.ntypes[src][0] == data.ntypes[src])
    # get the target type
    tgt = data.target_edge_index[1]
    tgt_type = data.ntypes[tgt][0]
    assert torch.all(data.ntypes[tgt][0] == data.ntypes[tgt])

    edge_type = data.target_edge_type[0]

    # from edges that are of the same source and target types, sample negative edges.
    src_index = (data.ntypes == src_type).nonzero().squeeze()
    tgt_index = (data.ntypes == tgt_type).nonzero().squeeze()
    
    cand_edge_index = torch.cartesian_prod(src_index, tgt_index).T
    _, num_match = edge_match(data.target_edge_index, cand_edge_index)
    neg_edge_index = cand_edge_index[:, num_match==0]
    shuffle = torch.randperm(neg_edge_index.size(1))
    neg_edge_index = neg_edge_index[:, shuffle][:, :num_neg_samples]

    neg_edges = torch.cat([neg_edge_index,
                           edge_type.repeat(num_neg_samples).unsqueeze(0), 
                            torch.zeros(neg_edge_index.size(1),
                                        device=neg_edge_index.device,dtype=torch.int64).unsqueeze(0)]).T

    return neg_edges