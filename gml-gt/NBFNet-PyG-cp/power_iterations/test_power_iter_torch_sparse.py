import torch
from tqdm import tqdm
from pprint import pprint
from torch_sparse import spspmm

"""
The following code does not work! since torch_sparse.spspmm cannot compute gradients.
"""


"""
Here we have two graphs, one with node_id 0 - 3, and another with node_id 4 - 7.
We are interested in the set of all paths (upto length max_path_length) from node 0 to 2 in graph1, and node 4 to 7 in graph2.
For the edges that are inside these paths, we would like the optimization to give it a high edge mask value.
For the edges that are outside these paths, this optimization will not change its weight.
"""
EPS = 1e-15
device = torch.device("cuda:3")
max_path_length = 3
edge_index = torch.tensor(
    [[0, 0, 1, 1, 3, 4, 5, 5, 6], [1, 3, 2, 3, 2, 5, 7, 6, 4]],
    dtype=torch.int64,
    device=device,
)
num_nodes = torch.max(edge_index) + 1

head_index = torch.tensor([0, 4], dtype=torch.int64, device=device)
tail_index = torch.tensor([2, 7], dtype=torch.int64, device=device)
tail_batch_index = torch.tensor([0, 1], dtype=torch.int64, device=device)
edge_mask = torch.nn.Parameter(torch.randn(edge_index.size(1), device=device))
optimizer = torch.optim.Adam([edge_mask], lr=0.001)
losses = []
print("Starting Edge Weight:")
pprint(edge_index)
pprint(torch.sigmoid(edge_mask))
for e in tqdm(range(100)):
    optimizer.zero_grad()
    assert not torch.any(torch.isnan(edge_mask))
    # get mask
    mask = torch.sigmoid(edge_mask)
    # get the adj matrix
    # mask_adj = torch.sparse_coo_tensor(edge_index, mask, size=(num_nodes, num_nodes))
    mask_adj = torch.zeros((num_nodes, num_nodes), device=device)
    mask_adj[edge_index[0], edge_index[1]] = mask
    head_mask_adj = mask_adj[head_index]

    adj = torch.zeros_like(mask_adj)
    adj[edge_index[0], edge_index[1]] = torch.ones_like(mask)
    head_adj = adj[head_index]

    head_edge_index = head_mask_adj.nonzero().T
    head_mask = head_mask_adj[head_edge_index[0], head_edge_index[1]]
    # the power iteration
    loss_on_path = 0

    # initialize the sparse matrix M and A
    prev_edge_index = edge_index
    prev_mask = mask
    prev_adj = torch.ones_like(mask)
    for i in range(1, max_path_length + 1):

        # the 1 hop paths
        if i == 1:
            # the aggr. weight will simply be the head mask
            aggr_weight = head_mask
            # norm is also 1 since it is a directed graph (not multi graph)
            norm = torch.ones_like(head_mask)
            # path_index which records which tails have connection with head is also simply the head_edge_index
            path_index = head_edge_index

        # for l hop paths where l >= 2
        else:
            # for l hop paths where l >= 3
            if i != 2:
                # compute M^(l-1) by multiplying initial M to M^(l-2)
                new_edge_index_m, new_mask = spspmm(
                    prev_edge_index,
                    prev_mask,
                    edge_index,
                    mask,
                    m=num_nodes,
                    k=num_nodes,
                    n=num_nodes,
                    coalesced=True,
                )
                # compute A^(l-1) by multiplying initial A to A^(l-2)
                new_edge_index_a, new_adj = spspmm(
                    prev_edge_index,
                    prev_adj,
                    edge_index,
                    torch.ones_like(mask),
                    m=num_nodes,
                    k=num_nodes,
                    n=num_nodes,
                    coalesced=True,
                )
                assert torch.equal(new_edge_index_m, new_edge_index_a)
                # update M^(l-2) to M^(l-1) and A^(l-2) to A^(l-1)
                prev_edge_index = new_edge_index_m
                prev_mask = new_mask
                prev_adj = new_adj

            # get the aggr weight of all l-hop paths from head to all possible tails
            # calculate the product of u @ M^(l-1)
            path_index_w, aggr_weight = spspmm(
                head_edge_index,
                head_mask,
                prev_edge_index,
                prev_mask,
                m=head_index.size(0),
                n=num_nodes,
                k=num_nodes,
                coalesced=True,
            )
            # calculate the number of all l-hop paths from head to all possible tails
            # calculate the product of u @ A^(l-1)
            path_index_n, norm = spspmm(
                head_edge_index,
                torch.ones_like(head_mask),
                prev_edge_index,
                prev_adj,
                m=head_index.size(0),
                n=num_nodes,
                k=num_nodes,
                coalesced=True,
            )

            # the path_index, which stores the indices where there is a l-hop path from head to tail should match
            assert torch.equal(path_index_w, path_index_n)
            path_index = path_index_w

        # in sparse tensor we only store the values where there is a path, so the normalization (# paths) should be positive.
        assert torch.all(norm > 0)
        # normalize the aggr weight by the number of paths
        norm_weight = torch.pow(aggr_weight / norm, 1 / i)
        # create dense tensor to get the score for all the tails with the normalized weight
        weight = torch.zeros((head_index.size(0), num_nodes), device=device)
        weight[path_index[0], path_index[1]] = norm_weight
        # index into the tails of interest
        path_weight = weight[tail_batch_index, tail_index]
        # get the sum of the scores as the path_weight
        path_weight = path_weight.sum()
        # aggregate the loss for the l-hop paths
        loss_on_path += path_weight

    # get the average by normalizing it with (L-1)
    if max_path_length > 1:
        loss_on_path = 1 / (max_path_length - 1) * loss_on_path
    # get the negative log s.t. higher weight -> lower loss
    loss_on_path = -torch.log(loss_on_path + EPS)
    losses.append(loss_on_path.item())
    loss_on_path.backward()
    optimizer.step()
print("Finishing Edge Weight, the last two edges are not inside the set of paths:")
pprint(edge_index)
pprint(torch.sigmoid(edge_mask))
print("All Done")
