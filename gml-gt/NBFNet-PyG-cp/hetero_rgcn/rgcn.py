import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from hetero_rgcn.layer import RGCNConv, FastRGCNConv
import copy

class RGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, num_relation, num_nodes, aggr='mean',
                 use_node_emb = True, random_node_features='const', randomized_edge_drop=0.0, 
                 eval_on_edge_drop=False, remove_ground_truth=False, keep_ground_truth=False):
        super().__init__()
        self.dims = [input_dim] + list(hidden_dims)
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(FastRGCNConv(self.dims[i], self.dims[i+1], num_relation, aggr=aggr))
            # self.layers.append(RGCNConv(self.dims[i], self.dims[i+1], num_relation))

        # Node embs
        self.use_node_emb = use_node_emb
        self.random_node_features = random_node_features
        if self.use_node_emb:
            self.node_emb = Parameter(torch.empty(num_nodes, input_dim))


        self.hops = len(self.dims) - 1

        # random edge drops
        self.randomized_edge_drop = randomized_edge_drop
        self.eval_on_edge_drop = eval_on_edge_drop
        # ground truths
        self.remove_ground_truth = remove_ground_truth
        self.keep_ground_truth = keep_ground_truth

        self.reset_parameters()

    def reset_parameters(self):
        if self.use_node_emb:
            torch.nn.init.xavier_uniform_(self.node_emb)
    
    def forward(self, data, target_edge_index):
        if self.use_node_emb:
            x = self.node_emb[data.x]
        else:
            if self.random_node_features == 'const':
                x = torch.ones(data.x.size(0), self.dims[0], device=data.x.device)*0.1
            else:
                x = torch.randn(data.x.size(0), self.dims[0], device=data.x.device)

        for layer in self.layers:
            x = layer(x, data.edge_index, data.edge_type).relu_()

        src, tgt = target_edge_index

        score = x[src]*x[tgt]
        score = score.sum(dim=-1)

        return score
    
    def masking(self, data):
        if (data.split == 'train' or self.eval_on_edge_drop) and self.randomized_edge_drop > 0:
            random_edge_drop = torch.rand(data.edge_index.size(1)) > self.randomized_edge_drop
            data = copy.copy(data)
            data.edge_index = data.edge_index[:, random_edge_drop]
            data.edge_type = data.edge_type[random_edge_drop]
            data.edge_batch = data.edge_batch[random_edge_drop]

        return data