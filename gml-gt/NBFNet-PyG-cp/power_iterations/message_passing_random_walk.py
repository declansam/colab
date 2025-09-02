import torch
from tqdm import tqdm
from pprint import pprint
from torch_geometric.data import Data


EPS = 1e-15
device = torch.device("cuda:3")
max_path_length = 3
edge_index = torch.tensor(
    [[0, 0, 0, 1, 1, 2, 3, 4], [1, 2, 3, 0, 4, 0, 0, 1]],
    dtype=torch.int64,
    device=device,
)
edge_batch = torch.tensor(
    [0, 0, 0, 0, 0],
    dtype=torch.int64,
    device=device,
)
subgraph_num_nodes = torch.tensor([5], dtype=torch.int64, device=device)
central_node_index = torch.tensor([[0]], dtype=torch.int64, device=device)
mask = torch.ones_like(edge_batch)
