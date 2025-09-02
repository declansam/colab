from nbfnet import tasks
import torch

edge_index = torch.tensor([[0, 1], [0, 2], [1, 2]]).T
query_edge_index = torch.tensor([[0, 1]]).T


tasks.edge_match(edge_index, query_edge_index)
