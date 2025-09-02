import torch
from tqdm import tqdm
from pprint import pprint


def path_loss(
    edge_index,
    edge_batch,
    path_mask,
    subgraph_num_nodes,
    central_node_index,
    max_path_length,
):
    """Compute the path loss. This does it per subgraph since backprop using sparse matrix multiplication explodes the memory anyways."""
    EPS = 1e-15
    batch_size = subgraph_num_nodes.size(0)
    total_loss = 0
    for i in range(batch_size):
        # get the edges for the i-th subgraph
        num_nodes = subgraph_num_nodes[i]
        edge_mask = edge_batch == i
        s_edge_index = edge_index[:, edge_mask]
        s_path_mask = path_mask[edge_mask]
        # create the mask adj matrix
        mask_adj = torch.sparse_coo_tensor(
            s_edge_index, s_path_mask, size=(num_nodes, num_nodes)
        )
        mask_adj = mask_adj.coalesce()
        # the adjacency matrix
        adj = torch.sparse_coo_tensor(
            s_edge_index,
            torch.ones_like(s_path_mask),
            size=(num_nodes, num_nodes),
        )
        adj = adj.coalesce()

        # create dense matrix for head nodes
        s_head_index = central_node_index[:, 0][i]
        s_tail_index = central_node_index[:, 1:][i]
        s_tail_index = s_tail_index.flatten()
        # get the out_edges from the head nodes
        head_edge_mask = torch.isin(s_edge_index[0], s_head_index)
        head_edge_index = s_edge_index[:, head_edge_mask]
        head_path_mask = s_path_mask[head_edge_mask]
        # create the dense matrix for head nodes
        head_mask_adj = torch.zeros((1, mask_adj.size(1)), device=mask_adj.device)
        head_mask_adj[0, head_edge_index[1]] = head_path_mask
        head_adj = torch.zeros((1, mask_adj.size(1)), device=mask_adj.device)
        head_adj[0, head_edge_index[1]] = torch.ones_like(head_path_mask)

        # the power iteration
        loss_on_path = 0
        for i in range(1, max_path_length + 1):
            # the 1 hop paths
            if i == 1:
                aggr_weight = head_mask_adj
                norm = head_adj.clamp(min=1)
                weight = aggr_weight / norm
            else:
                if i != 2:
                    mask_adj = torch.sparse.mm(mask_adj, mask_adj)
                    adj = torch.sparse.mm(adj, adj)
                    """
                    mask_adj = torch.mm(mask_adj, mask_adj)
                    adj = torch.mm(adj, adj)
                    """

                aggr_weight = torch.sparse.mm(head_mask_adj, mask_adj)
                norm = torch.sparse.mm(head_adj, adj).clamp(min=1)
                """
                aggr_weight = torch.mm(head_mask_adj, mask_adj)
                norm = torch.mm(head_adj, adj).clamp(min=1)
                """

                weight = torch.pow(aggr_weight / norm + EPS, 1 / i)

            path_weight = weight[0, s_tail_index]
            path_weight = path_weight.sum()
            loss_on_path += path_weight
        total_loss += loss_on_path

    if max_path_length > 1:
        total_loss = 1 / (max_path_length - 1) * total_loss
    total_loss = -torch.log(total_loss + EPS)

    return total_loss


EPS = 1e-15
device = torch.device("cuda:3")
max_path_length = 3
edge_index = torch.tensor(
    [[0, 0, 1, 1, 3, 3, 0, 1, 1, 1, 2], [1, 3, 2, 3, 2, 4, 1, 3, 4, 2, 0]],
    dtype=torch.int64,
    device=device,
)
edge_batch = torch.tensor(
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    dtype=torch.int64,
    device=device,
)
subgraph_num_nodes = torch.tensor([5, 5], dtype=torch.int64, device=device)
central_node_index = torch.tensor(
    [[0, 2, 4], [0, 3, 4]], dtype=torch.int64, device=device
)

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
    loss_on_path = path_loss(
        edge_index,
        edge_batch,
        mask,
        subgraph_num_nodes,
        central_node_index,
        max_path_length,
    )
    losses.append(loss_on_path.item())
    loss_on_path.backward()
    optimizer.step()

print("Finishing Edge Weight, the last two edges are not inside the set of paths:")
pprint(edge_index)
pprint(torch.sigmoid(edge_mask))
