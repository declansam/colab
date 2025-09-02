from torch_geometric.utils import k_hop_subgraph
import torch

edge_index = torch.tensor([[1, 2, 3],
                           [0, 1, 1]])

subset, edge_index, mapping, edge_mask = k_hop_subgraph(0, 2, edge_index, relabel_nodes=False, directed=True)


print(subset)
print(edge_index)
print(mapping)
print(edge_mask)


'''
Expected Outcome:
>>> edge_index
tensor([[1, 2, 3],
        [0, 1, 1]])

>>> edge_mask
tensor([True,  True,  True])


Actual Outcome:
>>> edge_index
tensor([[2, 3],
        [1, 1]])
>>> edge_mask
tensor([False,  True,  True])
'''