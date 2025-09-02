import copy
from collections.abc import Sequence

import torch
from torch import nn, autograd

from torch_scatter import scatter_add
import torch_geometric
from torch_geometric.utils import k_hop_subgraph


pyg_version = [int(i) for i in torch_geometric.__version__.split(".")]
if pyg_version[1] == 0:
    from . import layers
elif pyg_version[1] == 4:
    from . import layers_updated as layers

from . import tasks
    

class NBFNet(nn.Module):

    def __init__(self, input_dim, hidden_dims, num_relation, message_func="distmult", aggregate_func="pna",
                 short_cut=False, layer_norm=False, activation="relu", concat_hidden=False, num_mlp_layer=2,
                 dependent=True, remove_one_hop=False, num_beam=10, path_topk=10, get_path = False, use_pyg_propagation = False, 
                 randomized_edge_drop=0.0, eval_on_edge_drop=False):
        super(NBFNet, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]

        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut  # whether to use residual connections between GNN layers
        self.concat_hidden = concat_hidden  # whether to compute final states as a function of all layer outputs or last
        self.remove_one_hop = remove_one_hop  # whether to dynamically remove one-hop edges from edge_index
        self.num_beam = num_beam
        self.path_topk = path_topk
        self.randomized_edge_drop = randomized_edge_drop
        self.eval_on_edge_drop = eval_on_edge_drop

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layers.GeneralizedRelationalConv(self.dims[i], self.dims[i + 1], num_relation,
                                                                self.dims[0], message_func, aggregate_func, layer_norm,
                                                                activation, dependent, use_pyg_propagation))

        feature_dim = (sum(hidden_dims) if concat_hidden else hidden_dims[-1]) + input_dim # this is the feature dim to the prediction head, which takes h & t

        # additional relation embedding which serves as an initial 'query' for the NBFNet forward pass
        # each layer has its own learnable relations matrix, so we send the total number of relations, too
        self.query = nn.Embedding(num_relation, input_dim)
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(num_mlp_layer - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)
        self.get_path = get_path

        self.hops = len(self.dims) - 1
        self.node_embedding_size = feature_dim
        self.rel_embedding_size = input_dim

        self.dynamic_explanation = False # whether we are doing dynamic explanation

    def remove_easy_edges(self, data, h_index, t_index, r_index=None):
        # we remove training edges (we need to predict them at training time) from the edge index
        # think of it as a dynamic edge dropout
        h_index_ext = torch.cat([h_index, t_index], dim=-1)
        t_index_ext = torch.cat([t_index, h_index], dim=-1)
        inv_rels = r_index >= self.num_relation//2 # create a mask where there are inv_rels

        # Compute the inverse relations using vectorized operations
        r_ext = r_index.clone()
        r_ext[inv_rels] -= self.num_relation // 2
        r_ext[~inv_rels] += self.num_relation // 2

        r_index_ext = torch.cat([r_index, r_ext], dim=-1)

        # r_index_ext = torch.cat([r_index, r_index + self.num_relation // 2], dim=-1)
        if self.remove_one_hop and not self.get_path:
            # we remove all existing immediate edges between heads and tails in the batch
            edge_index = data.edge_index
            easy_edge = torch.stack([h_index_ext, t_index_ext]).flatten(1)
            index = tasks.edge_match(edge_index, easy_edge)[0]
            mask = ~index_to_mask(index, data.num_edges)
        else:
            # we remove existing immediate edges between heads and tails in the batch with the given relation
            edge_index = torch.cat([data.edge_index, data.edge_type.unsqueeze(0)])
            # note that here we add relation types r_index_ext to the matching query
            easy_edge = torch.stack([h_index_ext, t_index_ext, r_index_ext]).flatten(1)
            index = tasks.edge_match(edge_index, easy_edge)[0]
            mask = ~index_to_mask(index, data.num_edges)

        data = copy.copy(data)
        data.edge_index = data.edge_index[:, mask]
        data.edge_type = data.edge_type[mask]
        return data

    def negative_sample_to_tail(self, h_index, t_index, r_index, mode):
        # convert p(h | t, r) to p(t' | h', r')
        # h' = t, r' = r^{-1}, t' = h
        if mode is not None:
            is_t_neg = mode.unsqueeze(1).to(torch.bool)
        else:
            is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True) # if the head index changes, it knows that it's head-batch, True indicates Tail Batch
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation // 2)
        return new_h_index, new_t_index, new_r_index

    def bellmanford(self, data, h_index, r_index, separate_grad=False, edge_weight=None, preserved_edges=None, drop_edges=False):
        batch_size = len(r_index)

        # initialize queries (relation types of the given triples)
        query = self.query(r_index)
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        size = (data.num_nodes, data.num_nodes)
        
        if edge_weight is None:
            edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()
            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight, preserved_edges, drop_edges)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def masking_and_conversion_to_tail_prediction(self, data, batch, triple=None, mode=None):
        h_index, t_index, r_index = batch.unbind(-1) #(batch_size, num_triples)

        if data.split == 'train':
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        if (self.get_path and data.split == 'train') and triple is not None: # when getting paths we mask the true triple
            h, t, r = triple.unbind(-1)
            data = self.remove_easy_edges(data, h, t, r)

        if (data.split == 'train' or self.eval_on_edge_drop) and self.randomized_edge_drop > 0:
            random_edge_drop = torch.rand(data.edge_index.size(1)) > self.randomized_edge_drop
            data = copy.copy(data)
            data.edge_index = data.edge_index[:, random_edge_drop]
            data.edge_type = data.edge_type[random_edge_drop]
        # else:
        #     random_edge_drop = torch.ones(data.edge_index.size(1)).to(torch.bool)

        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, mode)
        if mode is None: # we have some padded index in the explainer case
            assert (h_index[:, [0]] == h_index).all()
            assert (r_index[:, [0]] == r_index).all()

        batch = torch.stack([h_index, t_index, r_index], dim=-1)
        return data, batch

    def forward(self, data, batch, triple = None, edge_weight=None, mode=None, return_emb=False, preserved_edges = None, drop_edges=False):
        '''
        data: the torch geometric data, holds the graph information.
        batch: the batch of triples to make the prediction.
        triple: the true triple, pass it in case we want to mask only the true triple (only applicable when doing get_path)
        edge_weight: the edge weight used for explanaton
        mode: used for static explainers
        return_emb: a flag to use when returning the emb, used for explainers
        preserved_edges: tensor indicating which edges were preserved by the explainer
        drop_edges: a flag to use when you are dropping the edges labeled as 0 on the edge_weight.
        '''
        # mask trivial paths & turn all triples in a batch into tail prediction mode
        if not self.dynamic_explanation: # if dynamic explanation, this will be done by the explainer
            data, batch = self.masking_and_conversion_to_tail_prediction(data, batch, triple, mode)

        h_index, t_index, r_index = batch.unbind(-1) #(batch_size, num_triples)

        shape = h_index.shape

        # edge_weight = torch.zeros(h_index.size(0), data.edge_index.size(1), device=h_index.device, dtype=torch.bool)
        # random_edge_drop = random_edge_drop.expand_as(edge_weight)
        # edge_weight[random_edge_drop] = 1
        # preserved_edges = edge_weight
        # drop_edges = True

        # message passing and updated node representations
        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0], edge_weight=edge_weight, preserved_edges=preserved_edges, drop_edges=drop_edges)  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        # extract representations of tail entities from the updated node states
        feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)

        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature).squeeze(-1)
        if return_emb: # Return the predicted score, Node embeddings, Relation Emb
            if self.dynamic_explanation: # for DynamicRelPGExplainer
                return score.view(shape).detach(), output["node_feature"].detach(), self.query.weight.detach()
            else: # for RelPGExplainer
                return score.view(shape).detach(), feature.detach(), self.query.weight.detach(), h_index[:, 0], r_index[:, 0]
        return score.view(shape)

    def visualize(self, data, batch):
        assert batch.shape == (1, 3)
        h_index, t_index, r_index = batch.unbind(-1)

        if self.get_path and data.split == 'train':
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        output = self.bellmanford(data, h_index, r_index, separate_grad=True)
        feature = output["node_feature"]
        edge_weights = output["edge_weights"]

        index = t_index.unsqueeze(0).unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(1, index).squeeze(0)
        score = self.mlp(feature).squeeze(-1)

        edge_grads = autograd.grad(score, edge_weights)
        distances, back_edges = self.beam_search_distance(data, edge_grads, h_index, t_index, self.num_beam)
        paths, weights = self.topk_average_length(distances, back_edges, t_index, self.path_topk)

        return paths, weights

    @torch.no_grad()
    def beam_search_distance(self, data, edge_grads, h_index, t_index, num_beam=10):
        # beam search the top-k distance from h to t (and to every other node)
        num_nodes = data.num_nodes
        input = torch.full((num_nodes, num_beam), float("-inf"), device=h_index.device)
        input[h_index, 0] = 0
        edge_mask = data.edge_index[0, :] != t_index

        distances = []
        back_edges = []
        for edge_grad in edge_grads:
            # we don't allow any path goes out of t once it arrives at t
            node_in, node_out = data.edge_index[:, edge_mask]
            relation = data.edge_type[edge_mask]
            edge_grad = edge_grad[edge_mask]

            message = input[node_in] + edge_grad.unsqueeze(-1) # (num_edges, num_beam)
            # (num_edges, num_beam, 3)
            msg_source = torch.stack([node_in, node_out, relation], dim=-1).unsqueeze(1).expand(-1, num_beam, -1)

            # (num_edges, num_beam)
            is_duplicate = torch.isclose(message.unsqueeze(-1), message.unsqueeze(-2)) & \
                           (msg_source.unsqueeze(-2) == msg_source.unsqueeze(-3)).all(dim=-1)
            # pick the first occurrence as the ranking in the previous node's beam
            # this makes deduplication easier later
            # and store it in msg_source
            is_duplicate = is_duplicate.float() - \
                           torch.arange(num_beam, dtype=torch.float, device=message.device) / (num_beam + 1)
            prev_rank = is_duplicate.argmax(dim=-1, keepdim=True)
            msg_source = torch.cat([msg_source, prev_rank], dim=-1) # (num_edges, num_beam, 4)

            node_out, order = node_out.sort()
            node_out_set = torch.unique(node_out)
            # sort messages w.r.t. node_out
            message = message[order].flatten() # (num_edges * num_beam)
            msg_source = msg_source[order].flatten(0, -2) # (num_edges * num_beam, 4)
            size = node_out.bincount(minlength=num_nodes)
            msg2out = size_to_index(size[node_out_set] * num_beam)
            # deduplicate messages that are from the same source and the same beam
            is_duplicate = (msg_source[1:] == msg_source[:-1]).all(dim=-1)
            is_duplicate = torch.cat([torch.zeros(1, dtype=torch.bool, device=message.device), is_duplicate])
            message = message[~is_duplicate]
            msg_source = msg_source[~is_duplicate]
            msg2out = msg2out[~is_duplicate]
            size = msg2out.bincount(minlength=len(node_out_set))

            if not torch.isinf(message).all():
                # take the topk messages from the neighborhood
                # distance: (len(node_out_set) * num_beam)
                distance, rel_index = scatter_topk(message, size, k=num_beam)
                abs_index = rel_index + (size.cumsum(0) - size).unsqueeze(-1)
                # store msg_source for backtracking
                back_edge = msg_source[abs_index] # (len(node_out_set) * num_beam, 4)
                distance = distance.view(len(node_out_set), num_beam)
                back_edge = back_edge.view(len(node_out_set), num_beam, 4)
                # scatter distance / back_edge back to all nodes
                distance = scatter_add(distance, node_out_set, dim=0, dim_size=num_nodes) # (num_nodes, num_beam)
                back_edge = scatter_add(back_edge, node_out_set, dim=0, dim_size=num_nodes) # (num_nodes, num_beam, 4)
            else:
                distance = torch.full((num_nodes, num_beam), float("-inf"), device=message.device)
                back_edge = torch.zeros(num_nodes, num_beam, 4, dtype=torch.long, device=message.device)

            distances.append(distance)
            back_edges.append(back_edge)
            input = distance

        return distances, back_edges

    def topk_average_length(self, distances, back_edges, t_index, k=10):
        # backtrack distances and back_edges to generate the paths
        paths = []
        average_lengths = []

        for i in range(len(distances)):
            distance, order = distances[i][t_index].flatten(0, -1).sort(descending=True)
            back_edge = back_edges[i][t_index].flatten(0, -2)[order]
            for d, (h, t, r, prev_rank) in zip(distance[:k].tolist(), back_edge[:k].tolist()):
                if d == float("-inf"):
                    break
                path = [(h, t, r)]
                for j in range(i - 1, -1, -1):
                    h, t, r, prev_rank = back_edges[j][h, prev_rank].tolist()
                    path.append((h, t, r))
                paths.append(path[::-1])
                average_lengths.append(d / len(path))

        if paths:
            average_lengths, paths = zip(*sorted(zip(average_lengths, paths), reverse=True)[:k])

        return paths, average_lengths


def index_to_mask(index, size):
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask


def size_to_index(size):
    range = torch.arange(len(size), device=size.device)
    index2sample = range.repeat_interleave(size)
    return index2sample


def multi_slice_mask(starts, ends, length):
    values = torch.cat([torch.ones_like(starts), -torch.ones_like(ends)])
    slices = torch.cat([starts, ends])
    mask = scatter_add(values, slices, dim=0, dim_size=length + 1)[:-1]
    mask = mask.cumsum(0).bool()
    return mask


def scatter_extend(data, size, input, input_size):
    new_size = size + input_size
    new_cum_size = new_size.cumsum(0)
    new_data = torch.zeros(new_cum_size[-1], *data.shape[1:], dtype=data.dtype, device=data.device)
    starts = new_cum_size - new_size
    ends = starts + size
    index = multi_slice_mask(starts, ends, new_cum_size[-1])
    new_data[index] = data
    new_data[~index] = input
    return new_data, new_size


def scatter_topk(input, size, k, largest=True):
    index2graph = size_to_index(size)
    index2graph = index2graph.view([-1] + [1] * (input.ndim - 1))

    mask = ~torch.isinf(input)
    max = input[mask].max().item()
    min = input[mask].min().item()
    safe_input = input.clamp(2 * min - max, 2 * max - min)
    offset = (max - min) * 4
    if largest:
        offset = -offset
    input_ext = safe_input + offset * index2graph
    index_ext = input_ext.argsort(dim=0, descending=largest)
    num_actual = size.clamp(max=k)
    num_padding = k - num_actual
    starts = size.cumsum(0) - size
    ends = starts + num_actual
    mask = multi_slice_mask(starts, ends, len(index_ext)).nonzero().flatten()

    if (num_padding > 0).any():
        # special case: size < k, pad with the last valid index
        padding = ends - 1
        padding2graph = size_to_index(num_padding)
        mask = scatter_extend(mask, num_actual, padding[padding2graph], num_padding)[0]

    index = index_ext[mask] # (N * k, ...)
    value = input.gather(0, index)
    if isinstance(k, torch.Tensor) and k.shape == size.shape:
        value = value.view(-1, *input.shape[1:])
        index = index.view(-1, *input.shape[1:])
        index = index - (size.cumsum(0) - size).repeat_interleave(k).view([-1] + [1] * (index.ndim - 1))
    else:
        value = value.view(-1, k, *input.shape[1:])
        index = index.view(-1, k, *input.shape[1:])
        index = index - (size.cumsum(0) - size).view([-1] + [1] * (index.ndim - 1))

    return value, index