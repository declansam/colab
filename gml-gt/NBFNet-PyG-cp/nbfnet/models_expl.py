import logging
import copy
from collections.abc import Sequence
import time
import torch
from torch import nn, autograd

from torch_scatter import scatter_add, scatter
import torch_geometric
from torch_geometric.utils import k_hop_subgraph
from explainers.data_util import recreate_data_object

from torch.distributions import LogNormal
from scipy.stats import beta

pyg_version = [int(i) for i in torch_geometric.__version__.split(".")]
if pyg_version[1] == 0:
    from . import layers
elif pyg_version[1] >= 4:
    from . import layers_expl as layers

from . import tasks

logger = logging.getLogger(__file__)


class NBFNet(nn.Module):
    '''
    input_dim,                   # Dimension of input node features
    hidden_dims,                 # Hidden layer dimensions (can be list or single int)
    num_relation,                # Total number of relation types
    message_func="distmult",     # Message function (distmult/transe)
    aggregate_func="pna",        # Aggregation function
    short_cut=False,             # Residual connections
    layer_norm=False,            # Layer normalization
    activation="relu",           # Activation function
    concat_hidden=False,         # Concatenate all layer outputs
    num_mlp_layer=2,             # MLP layers for final prediction
    dependent=True,              # Whether to use dependent relations
    remove_one_hop=False,        # Whether to dynamically remove one-hop edges from edge_index
    num_beam=10,                 # Number of beam search for distance
    path_topk=10,                # Top-k paths for explanation
    get_path=False,              # Whether to get paths
    use_pyg_propagation=True,    # Whether to use pyg propagation
    randomized_edge_drop=0.0,    # Probability to randomly drop an edge
    rw_dropout=False,            # Whether to do random walk dropout
    distance_dropout=False,      # Whether to drop edges based on disance
    max_edge_drop_prob=0.9,      # Max dropout prob an edge can have
    max_dropout_distance=-1,     # Max distance we are considering
    eval_dropout_distance=-1,    # Min distance of edges to be preserved for evaluation
    eval_on_edge_drop=False,     # Whether to eval on edge drop
    remove_ground_truth=False,   # Whether to remove ground truth
    keep_ground_truth=False,     # Whether to keep the ground truth

    '''

    def __init__(
        self,
        input_dim,
        hidden_dims,
        num_relation,
        message_func="distmult",
        aggregate_func="pna",
        short_cut=False,
        layer_norm=False,
        activation="relu",
        concat_hidden=False,
        num_mlp_layer=2,
        dependent=True,
        remove_one_hop=False,
        num_beam=10,
        path_topk=10,
        get_path=False,
        use_pyg_propagation=True,
        randomized_edge_drop=0.0,
        rw_dropout=False,
        distance_dropout=False,
        max_edge_drop_prob=0.9,
        max_dropout_distance=-1,
        eval_dropout_distance=-1,
        eval_on_edge_drop=False,
        remove_ground_truth=False,
        keep_ground_truth=False,
    ):
        super(NBFNet, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]

        # Adding an entry (input_dim) to the hidden_dims list to create the dims list
        self.dims = [input_dim] + list(hidden_dims)

        # Number of relation types in the dataset
        self.num_relation = num_relation
        self.short_cut = (
            short_cut  # whether to use residual connections between GNN layers
        )
        self.concat_hidden = concat_hidden  # whether to compute final states as a function of all layer outputs or last
        self.remove_one_hop = remove_one_hop  # whether to dynamically remove one-hop edges from edge_index
        self.num_beam = num_beam
        self.path_topk = path_topk
        self.randomized_edge_drop = (
            randomized_edge_drop  # probability to randomly drop an edge
        )

        # * Edge Dropout Configs *
        self.rw_dropout = rw_dropout  # whether to do random walk dropout
        # whether to drop edges based on disance
        self.distance_dropout = distance_dropout
        # for distance based dropout, what is the max dropout prob an edge can have
        self.max_edge_drop_prob = max_edge_drop_prob
        # what is the max distance we are considering
        self.max_dropout_distance = max_dropout_distance
        # the min distance of edges to be preserved for evaluation
        self.eval_dropout_distance = eval_dropout_distance
        # whether to eval on edge drop
        self.eval_on_edge_drop = eval_on_edge_drop

        self.remove_ground_truth = remove_ground_truth  # whether to remove ground truth
        self.remove_ground_truth_control = False  # whether to remove the control
        self.keep_ground_truth = keep_ground_truth  # whether to keep the ground truth
        self.keep_ground_truth_control = False  # whether to keep the control
        assert use_pyg_propagation  # pyg propagation must be used to train & evaluate

        # * GNN Layers *
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i],
                    self.dims[i + 1],
                    num_relation,
                    self.dims[0],
                    message_func,
                    aggregate_func,
                    layer_norm,
                    activation,
                    dependent,
                    use_pyg_propagation,
                )
            )

        feature_dim = (
            sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        ) + input_dim  # this is the feature dim to the prediction head, which takes h & t

        # additional relation embedding which serves as an initial 'query' for the NBFNet forward pass
        # each layer has its own learnable relations matrix, so we send the total number of relations, too
        # These serve as queries that initialize the Bellman-Ford iteration.
        self.query = nn.Embedding(num_relation, input_dim)

        # MLP for final prediction
        # After message passing, the final node representations are passed into an MLP to predict scores for candidate tail entities.
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

    def mask_edges(self, data, mask):
        data = copy.copy(data)
        if hasattr(data, "edge_distance"):
            edge_distance = data.edge_distance[mask]
        else:
            edge_distance = None
        if hasattr(data, "edge_filter"):
            data = recreate_data_object(
                data,
                data.edge_index[:, mask],
                data.edge_type[mask],
                data.edge_batch[mask],
                edge_distance,
            )
        else:
            data.edge_index = data.edge_index[:, mask]
            data.edge_type = data.edge_type[mask]
            data.edge_batch = data.edge_batch[mask]
            if hasattr(data, "edge_distance"):
                data.edge_distance = data.edge_distance[mask]
            # ensure if no edge the num_edge will be 0
            index, counts = torch.unique(data.edge_batch, return_counts=True)
            subgraph_num_edges = torch.zeros_like(data.subgraph_num_edges)
            subgraph_num_edges[index] = counts
            data.subgraph_num_edges = subgraph_num_edges
        return data

    def remove_edges(self, data, edge_index, easy_edge):
        index, num_match = tasks.edge_match(edge_index, easy_edge)
        invalid_edges = torch.any(easy_edge == -1, dim=0)
        assert torch.all(num_match[invalid_edges] == 0)
        mask = ~index_to_mask(index, data.num_edges)
        data = self.mask_edges(data, mask)
        return data

    def remove_easy_edges(self, data, h_index, t_index, r_index=None):
        # we remove training edges (we need to predict them at training time) from the edge index
        # think of it as a dynamic edge dropout
        h_index_ext = torch.cat([h_index, t_index], dim=-1)
        t_index_ext = torch.cat([t_index, h_index], dim=-1)
        # create a mask where there are inv_rels
        inv_rels = r_index >= self.num_relation // 2

        # Compute the inverse relations using vectorized operations
        r_ext = r_index.clone()
        r_ext[inv_rels] -= self.num_relation // 2
        r_ext[~inv_rels] += self.num_relation // 2

        r_index_ext = torch.cat([r_index, r_ext], dim=-1)

        # r_index_ext = torch.cat([r_index, r_index + self.num_relation // 2], dim=-1)
        if self.remove_one_hop and not self.get_path:
            # we remove all existing immediate edges between heads and tails in the batch
            edge_index = torch.cat([data.edge_index, data.edge_batch.unsqueeze(0)])
            batch_id = (
                torch.arange(h_index.size(0), device=h_index.device)
                .unsqueeze(1)
                .expand_as(h_index_ext)
            )
            easy_edge = torch.stack([h_index_ext, t_index_ext, batch_id]).flatten(1)
            return self.remove_edges(data, edge_index, easy_edge)
        else:
            # we remove existing immediate edges between heads and tails in the batch with the given relation
            edge_index = torch.cat(
                [
                    data.edge_index,
                    data.edge_type.unsqueeze(0),
                    data.edge_batch.unsqueeze(0),
                ]
            )
            # note that here we add relation types r_index_ext to the matching query
            batch_id = (
                torch.arange(h_index.size(0), device=h_index.device)
                .unsqueeze(1)
                .expand_as(h_index_ext)
            )
            easy_edge = torch.stack(
                [h_index_ext, t_index_ext, r_index_ext, batch_id]
            ).flatten(1)
            return self.remove_edges(data, edge_index, easy_edge)

    def bellmanford(
        self, data, h_index, r_index, separate_grad=False, edge_weight=None
    ):
        # start = time.time()
        batch_size = len(r_index)

        # initialize queries (relation types of the given triples)
        # i.e. simply lookup the relation embeddings for the RELATION types of the triples in the batch
        # query.shape => (batch_size, input_dim): (16, 32)
        query = self.query(r_index)

        # x.expand_as(y) => expand tensor x to the same shape as y
        # expand_as can be done on singleton only i.e. [16, 1] -> [16, 32] BUT NOT [16, 2] -> [16, 32]
        # h_index.unsqueeze(-1) => [16, 1]
        # index.shape => (batch_size, input_dim): (16, 32)
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        # boundary.shape => (batch_size, data.num_nodes, input_dim): (16, 41105, 32)
        # Each of the 16 queries gets its own "graph state"
        # Each node in each graph has a 32-dimensional feature vector (initially zeros)
        boundary = torch.zeros(
            batch_size, data.max_num_nodes, self.dims[0], device=h_index.device
        )

        # scatter_add_(dim, index, src)
        # For each position in index, add values from src into self along the given dim.”
        #$ This sets the initial state of the head nodes to be the query embedding while all other nodes remain zero. 
        #$ This is a crucial step that localizes the message passing process to the head entities.
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        # boundary = boundary.scatter_add(1, index.unsqueeze(1), query.unsqueeze(1))

        #$ size of the adjacency matrix for a single graph
        size = (data.max_num_nodes, data.max_num_nodes)

        #$ list to store the hidden states of each layer and edge weights
        hiddens = []
        edge_weights = []
        layer_input = boundary

        #$ When training NBFNet, we have multiple subgraphs of different sizes in a batch, 
        #$ but GPUs need fixed-size tensors for efficient processing.
        #$ shape: [16, 173668] <-- ( batch_size, max(data.subgraph_num_edges) )
        '''
        # Query 1: "What is connected to dog via hypernym?"
        Subgraph 1: 3 edges
        (dog → animal), (dog → mammal), (animal → creature)

        # Query 2: "What is connected to car via hypernym?"  
        Subgraph 2: 5 edges
        (car → vehicle), (car → object), (vehicle → thing), (object → entity), (thing → concept)

        # Query 3: "What is connected to bird via hypernym?"
        Subgraph 3: 2 edges
        (bird → animal), (animal → creature)



        Subgraph sizes: [3, 5, 2]
        Max size: 5 


        batched_edges = [
            # Subgraph 1: 3 real edges + 2 padding
            [(dog,animal), (dog,mammal), (animal,creature), (0,0), (0,0)],
            
            # Subgraph 2: 5 real edges + 0 padding  
            [(car,vehicle), (car,object), (vehicle,thing), (object,entity), (thing,concept)],
            
            # Subgraph 3: 2 real edges + 3 padding
            [(bird,animal), (animal,creature), (0,0), (0,0), (0,0)]
        ]

        edge_filter = [
            [True,  True,  True,  False, False],  # Subgraph 1: first 3 are real
            [True,  True,  True,  True,  True ],  # Subgraph 2: all 5 are real
            [True,  True,  False, False, False]   # Subgraph 3: first 2 are real
        ]

        NOTE: 
        - edge_filter is a mask that indicates which edges are real (True) and which are padded (False).
        - done in file: explainers/data_util.py by create_batched_data() function
        '''
        if edge_weight is None:
            edge_weight = data.edge_filter

        # Loop through each layer of the NBFNet (GeneralizedRelationalConv)
        for layer in self.layers:
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()
            
            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            #$ This is the message-passing step. 
            # It calls the forward method of the current GeneralizedRelationalConv layer. 
            # It passes the current node states (layer_input), the original query embeddings (query), the initial boundary condition (boundary), 
            # and various graph-related tensors from the data object.
            #  The layer then performs aggregation, updating the node representations based on messages from their neighbors.
            hidden = layer(
                layer_input,
                query,
                boundary,
                data.edge_index,
                data.edge_type,
                size,
                data.batched_edge_index,
                data.batched_edge_type,
                data.edge_filter,
                edge_weight,
            )

            #$ residual connection here
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input

            #$ The newly computed hidden states and edge weights are stored, and 
            #$ layer_input is updated for the next iteration of the loop.
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        #$ After the loop, the original query embeddings are prepared to be concatenated with the final node features. 
        node_query = query.unsqueeze(1).expand(
            -1, data.max_num_nodes, -1
        )  # (batch_size, num_nodes, input_dim)

        #$ concatenates the hidden states from all layers (hiddens) with the node_query embeddings.
        #$ This creates a very expressive but high-dimensional feature vector.
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)

        #$ Otherwise, it only uses the hidden states from the last layer (hiddens[-1]) and concatenates them with the node_query.
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        # end = time.time()
        # logger.warning(f"* Bellman Ford Took {(end - start):.2f}s*")
        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def rw_edge_dropout(self, data, batch, randomwalker):
        randomwalker.node_dist = None  # reset the node distribution
        node_dist = randomwalker.train_mask(data, batch, return_node_dist=True)
        unique_edges = data.edge_index + data.edge_batch * data.num_nodes
        torch.save((data, batch, node_dist, unique_edges), "node_dists.pt")
        # get the prob of edge dropping based on the node_dist
        src = unique_edges[0]
        tgt = unique_edges[1]

        edge_prob = node_dist[src] * node_dist[tgt]
        mu = edge_prob.mean()
        var = edge_prob.var()
        a = ((1 - mu) / var - 1 / mu) * mu**2
        b = a * (1 / mu - 1)
        edge_imp = beta.cdf(edge_prob.to("cpu"), a=a.item(), b=b.item())

        # get the edges that have score of 0
        zero_mask = node_dist[src] * node_dist[tgt] == 0

        epsilon = 1e-12
        src_node_prob = torch.log(node_dist[src] + epsilon)
        tgt_node_prob = torch.log(node_dist[tgt] + epsilon)
        edge_prob = src_node_prob + tgt_node_prob  # equiv to log(src_prob * tgt_prob)
        edge_prob = -edge_prob  # flip it so that it gets positive values
        mean = torch.mean(edge_prob)
        std = torch.std(edge_prob)
        dist = LogNormal(loc=mean, scale=std)
        cdf_values = dist.cdf(edge_prob)
        edge_prob = 1 - cdf_values  # get the survival function.

        ...
        print("Done")

    def get_control_random_edges(self, data, edge_index):
        """Gets random edges that is of the same size as the ground truth"""
        assert torch.all(data.subgraph_num_edges == data.subgraph_num_edges[0])
        gt_edge_batch, count = torch.unique(data.gt_edge_batch, return_counts=True)

        argsort = torch.randn(data.edge_filter.shape, device=data.edge_filter.device)
        argsort = torch.argsort(argsort, dim=1)
        offset = torch.cumsum(data.subgraph_num_edges, 0) - data.subgraph_num_edges
        argsort += offset.unsqueeze(1)

        indices = torch.arange(
            data.edge_filter.size(1), device=data.edge_filter.device
        ).expand_as(data.edge_filter)
        mask = indices < count.unsqueeze(1)

        random_edge_id = argsort[mask]
        random_edges = edge_index[:, random_edge_id]
        return random_edges

    def masking(self, data, batch, randomwalker=None):
        h_index, t_index, r_index = batch.unbind(-1)
        if data.split == "train":
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        # remove the ground truth edges
        if self.remove_ground_truth:
            edge_index = torch.cat(
                [
                    data.edge_index,
                    data.edge_type.unsqueeze(0),
                    data.edge_batch.unsqueeze(0),
                ]
            )
            gt_edges = torch.cat(
                [
                    data.gt_edge_index,
                    data.gt_edge_type.unsqueeze(0),
                    data.gt_edge_batch.unsqueeze(0),
                ]
            )
            data = self.remove_edges(data, edge_index, gt_edges)

        # remove random edges that is not the ground truth
        if self.remove_ground_truth_control:
            edge_index = torch.cat(
                [
                    data.edge_index,
                    data.edge_type.unsqueeze(0),
                    data.edge_batch.unsqueeze(0),
                ]
            )
            random_edges = self.get_control_random_edges(data, edge_index)
            data = self.remove_edges(data, edge_index, random_edges)

        if (data.split == "train" or self.eval_on_edge_drop) and (
            self.randomized_edge_drop > 0
        ):

            if self.keep_ground_truth:
                # remove the ground truth and add it back later
                edge_index = torch.cat(
                    [
                        data.edge_index,
                        data.edge_type.unsqueeze(0),
                        data.edge_batch.unsqueeze(0),
                    ]
                )
                gt_edges = torch.cat(
                    [
                        data.gt_edge_index,
                        data.gt_edge_type.unsqueeze(0),
                        data.gt_edge_batch.unsqueeze(0),
                    ]
                )
                index, num_match = tasks.edge_match(edge_index, gt_edges)
                keep_edge_index = data.edge_index[:, index]
                keep_edge_type = data.edge_type[index]
                keep_edge_batch = data.edge_batch[index]
                data = self.remove_edges(data, edge_index, gt_edges)

            elif self.keep_ground_truth_control:
                edge_index = torch.cat(
                    [
                        data.edge_index,
                        data.edge_type.unsqueeze(0),
                        data.edge_batch.unsqueeze(0),
                    ]
                )
                random_edges = self.get_control_random_edges(data, edge_index)
                keep_edge_index = random_edges[:2]
                keep_edge_type = random_edges[2]
                keep_edge_batch = random_edges[3]
                data = self.remove_edges(data, edge_index, random_edges)

            else:
                keep_edge_index, keep_edge_type, keep_edge_batch = None, None, None

            if randomwalker is not None and self.rw_dropout:
                data = self.rw_edge_dropout(data, batch, randomwalker)
                raise NotImplementedError

            elif self.randomized_edge_drop > 0 and not self.distance_dropout:
                random_edge_drop = (
                    torch.rand(data.edge_index.size(1)) > self.randomized_edge_drop
                )
                data = self.mask_edges(data, random_edge_drop)

            if keep_edge_index is not None:
                # add back the removed ground truth / control edges
                edge_index = torch.cat((data.edge_index, keep_edge_index), dim=1)
                edge_type = torch.cat((data.edge_type, keep_edge_type))
                edge_batch = torch.cat((data.edge_batch, keep_edge_batch))
                # the edges need to be sorted based on the edge batch
                edge_batch, indices = torch.sort(edge_batch)
                edge_type = edge_type[indices]
                edge_index = edge_index[:, indices]
                data = copy.copy(data)
                if hasattr(data, "edge_filter"):
                    data = recreate_data_object(data, edge_index, edge_type, edge_batch)
                else:
                    data.edge_index = edge_index
                    data.edge_type = edge_type
                    data.edge_batch = edge_batch

                    index, counts = torch.unique(
                        data.edge_batch, return_counts=True
                    )  # ensure if no edge the num_edge will be 0
                    subgraph_num_edges = torch.zeros_like(data.subgraph_num_edges)
                    subgraph_num_edges[index] = counts
                    data.subgraph_num_edges = subgraph_num_edges

        return data

    def forward(self, data, batch, edge_weight=None, return_emb=False):
        """
        data: the torch geometric data, holds the graph information.
        batch: the batch of triples to make the prediction.
        edge_weight: the edge weight used for explanaton
        return_emb: a flag to use when returning the emb, used for explainers
        degree_by_weights: whether to calculate the degree information during aggregation using edge weights
        """

        # batch.shape == (batch_size, num_negative + 1, 3)
        # now h/t/r_index.shape == (batch_size, num_negative + 1)
        h_index, t_index, r_index = batch.unbind(-1)  # (batch_size, num_triples)
        
        # assumption: the batch has been converted to tail batch
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # (batch_size, num_negative + 1)
        shape = h_index.shape

        # message passing and updated node representations
        # h_index[:, 0] is the head index of the first triple in the batch
        # it's all the true heads in the batch
        # from [16, 33] h_index, we are feeding [16] h_index which is the true head of the triples in the batch
        # GOAL: Performs the core message-passing operation. 
        # It takes the first head and relation from each batch entry and propagates information through the graph (data). 
        # The result is a dictionary output containing the final node representations.
        output = self.bellmanford(
            data, h_index[:, 0], r_index[:, 0], edge_weight=edge_weight
        )
        feature = output["node_feature"]  # (batch_size, max_num_nodes, feature_dim)

        # some tails indicies has been marked as -1 to indicate that there are no negative tails in the ego-network
        invalid_tails = t_index == -1
        t_index = torch.where(invalid_tails, torch.zeros_like(t_index), t_index)
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        
        # extract representations of tail entities from the updated node states
        # (batch_size, num_negative + 1, feature_dim)
        # It uses the prepared index tensor to extract the learned features for each tail candidate.
        feature = feature.gather(1, index)

        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature).squeeze(-1)
        if return_emb:  # Return the predicted score, Node embeddings, Relation Emb
            return score.view(shape), output["node_feature"], self.query.weight
        return score.view(shape)

    def visualize(self, data, batch):
        assert batch.shape == (1, 3)
        h_index, t_index, r_index = batch.unbind(-1)

        if self.get_path and data.split == "train":
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        output = self.bellmanford(data, h_index, r_index, separate_grad=True)
        feature = output["node_feature"]
        edge_weights = output["edge_weights"]

        index = t_index.unsqueeze(0).unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(1, index).squeeze(0)
        score = self.mlp(feature).squeeze(-1)

        edge_grads = autograd.grad(score, edge_weights)
        distances, back_edges = self.beam_search_distance(
            data, edge_grads, h_index, t_index, self.num_beam
        )
        paths, weights = self.topk_average_length(
            distances, back_edges, t_index, self.path_topk
        )

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

            message = input[node_in] + edge_grad.unsqueeze(-1)  # (num_edges, num_beam)
            # (num_edges, num_beam, 3)
            msg_source = (
                torch.stack([node_in, node_out, relation], dim=-1)
                .unsqueeze(1)
                .expand(-1, num_beam, -1)
            )

            # (num_edges, num_beam)
            is_duplicate = torch.isclose(
                message.unsqueeze(-1), message.unsqueeze(-2)
            ) & (msg_source.unsqueeze(-2) == msg_source.unsqueeze(-3)).all(dim=-1)
            # pick the first occurrence as the ranking in the previous node's beam
            # this makes deduplication easier later
            # and store it in msg_source
            is_duplicate = is_duplicate.float() - torch.arange(
                num_beam, dtype=torch.float, device=message.device
            ) / (num_beam + 1)
            prev_rank = is_duplicate.argmax(dim=-1, keepdim=True)
            msg_source = torch.cat(
                [msg_source, prev_rank], dim=-1
            )  # (num_edges, num_beam, 4)

            node_out, order = node_out.sort()
            node_out_set = torch.unique(node_out)
            # sort messages w.r.t. node_out
            message = message[order].flatten()  # (num_edges * num_beam)
            msg_source = msg_source[order].flatten(0, -2)  # (num_edges * num_beam, 4)
            size = node_out.bincount(minlength=num_nodes)
            msg2out = size_to_index(size[node_out_set] * num_beam)
            # deduplicate messages that are from the same source and the same beam
            is_duplicate = (msg_source[1:] == msg_source[:-1]).all(dim=-1)
            is_duplicate = torch.cat(
                [torch.zeros(1, dtype=torch.bool, device=message.device), is_duplicate]
            )
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
                back_edge = msg_source[abs_index]  # (len(node_out_set) * num_beam, 4)
                distance = distance.view(len(node_out_set), num_beam)
                back_edge = back_edge.view(len(node_out_set), num_beam, 4)
                # scatter distance / back_edge back to all nodes
                distance = scatter_add(
                    distance, node_out_set, dim=0, dim_size=num_nodes
                )  # (num_nodes, num_beam)
                back_edge = scatter_add(
                    back_edge, node_out_set, dim=0, dim_size=num_nodes
                )  # (num_nodes, num_beam, 4)
            else:
                distance = torch.full(
                    (num_nodes, num_beam), float("-inf"), device=message.device
                )
                back_edge = torch.zeros(
                    num_nodes, num_beam, 4, dtype=torch.long, device=message.device
                )

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
            for d, (h, t, r, prev_rank) in zip(
                distance[:k].tolist(), back_edge[:k].tolist()
            ):
                if d == float("-inf"):
                    break
                path = [(h, t, r)]
                for j in range(i - 1, -1, -1):
                    h, t, r, prev_rank = back_edges[j][h, prev_rank].tolist()
                    path.append((h, t, r))
                paths.append(path[::-1])
                average_lengths.append(d / len(path))

        if paths:
            average_lengths, paths = zip(
                *sorted(zip(average_lengths, paths), reverse=True)[:k]
            )

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
    new_data = torch.zeros(
        new_cum_size[-1], *data.shape[1:], dtype=data.dtype, device=data.device
    )
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

    index = index_ext[mask]  # (N * k, ...)
    value = input.gather(0, index)
    if isinstance(k, torch.Tensor) and k.shape == size.shape:
        value = value.view(-1, *input.shape[1:])
        index = index.view(-1, *input.shape[1:])
        index = index - (size.cumsum(0) - size).repeat_interleave(k).view(
            [-1] + [1] * (index.ndim - 1)
        )
    else:
        value = value.view(-1, k, *input.shape[1:])
        index = index.view(-1, k, *input.shape[1:])
        index = index - (size.cumsum(0) - size).view([-1] + [1] * (index.ndim - 1))

    return value, index
