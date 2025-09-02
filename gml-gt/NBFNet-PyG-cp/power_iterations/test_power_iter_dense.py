import torch
from tqdm import tqdm
from pprint import pprint
from torch_sparse import spspmm

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
for e in tqdm(range(1000)):
    optimizer.zero_grad()
    assert not torch.any(torch.isnan(edge_mask))
    # get mask
    mask = torch.sigmoid(edge_mask)
    # get the adj matrix
    mask_adj = torch.zeros((num_nodes, num_nodes), device=device)
    mask_adj[edge_index[0], edge_index[1]] = mask
    head_mask_adj = mask_adj[head_index]

    adj = torch.zeros_like(mask_adj)
    adj[edge_index[0], edge_index[1]] = torch.ones_like(mask)
    head_adj = adj[head_index]

    # the power iteration
    loss_on_path = 0
    for i in range(1, max_path_length + 1):
        if i == 1:
            aggr_weight = head_mask_adj
            norm = head_adj.clamp(min=1)
            weight = aggr_weight / norm
        else:
            if i != 2:
                mask_adj = torch.mm(mask_adj, mask_adj)
                adj = torch.mm(adj, adj)
            aggr_weight = torch.mm(head_mask_adj, mask_adj)
            norm = torch.mm(head_adj, adj).clamp(min=1)
            weight = torch.pow((aggr_weight / norm) + EPS, 1 / i)

        path_weight = weight[tail_batch_index, tail_index]
        path_weight = path_weight.sum()
        loss_on_path += path_weight

    if max_path_length > 1:
        loss_on_path = 1 / (max_path_length - 1) * loss_on_path
    loss_on_path = -torch.log(loss_on_path + EPS)
    losses.append(loss_on_path.item())
    loss_on_path.backward()
    optimizer.step()
print("Finishing Edge Weight, the last two edges are not inside the set of paths:")
pprint(edge_index)
pprint(torch.sigmoid(edge_mask))
