import torch
from torch import nn
from torch.nn import functional as F
import time
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
from explainers.BaseExplainer import BaseExplainer
from explainers.PowerLink import get_path_loss
from explainers.data_util import (
    recreate_data_object,
    counter_factual_edge_filter,
    convert_multigraph_to_digraph,
)
from explainers.mst import kruskal_mst_edges
from torch_scatter import scatter, scatter_softmax
from nbfnet.tasks import edge_match
import logging

logger = logging.getLogger(__file__)


def convert_to_undirected(G):
    """
    Convert a directed graph to an undirected graph.

    For every edge (u, v) in G:
    - If there is no reciprocal edge (v, u) in G, add (u, v) and its corresponding mask and edge_id.
    - If there is a reciprocal edge (v, u) in G, compare the mask value and add the higher edge's mask and edge_id.

    Parameters:
    G (nx.DiGraph): The input directed graph.

    Returns:
    nx.Graph: The resulting undirected graph.
    """

    # Create an empty undirected graph
    undirected_G = nx.Graph()

    processed = []
    # Iterate over all edges in the directed graph
    for u, v in G.edges():
        if (u, v) in processed:
            continue
        # Check if there is a reciprocal edge
        if (v, u) in G.edges():
            # Compare the mask values and choose the higher one
            if G[u][v]["mask"] > G[v][u]["mask"]:
                mask = G[u][v]["mask"]
                edge_id = G[u][v]["edge_id"]
                edge_batch = G[u][v]["edge_batch"]
            else:
                mask = G[v][u]["mask"]
                edge_id = G[v][u]["edge_id"]
                edge_batch = G[v][u]["edge_batch"]
            processed.append((v, u))
        else:
            # If no reciprocal edge, use the current edge's attributes
            mask = G[u][v]["mask"]
            edge_id = G[u][v]["edge_id"]
            edge_batch = G[u][v]["edge_batch"]

        # Add the edge to the undirected graph
        undirected_G.add_edge(u, v, mask=mask, edge_id=edge_id, edge_batch=edge_batch)

    return undirected_G


def small_example(data, mask):
    # test with a small graph to see if it is working
    data.edge_index = torch.tensor(
        [[0, 0, 0, 1, 1, 2, 3, 4], [1, 2, 3, 0, 4, 0, 0, 1]],
        dtype=torch.int64,
        device=mask.device,
    )
    data.edge_batch = torch.tensor(
        [0, 0, 0, 0, 0, 0, 0, 0],
        dtype=torch.int64,
        device=mask.device,
    )
    data.subgraph_num_nodes = torch.tensor([5], dtype=torch.int64, device=mask.device)
    data.central_node_index = torch.tensor([[0]], dtype=torch.int64, device=mask.device)
    mask = torch.ones_like(data.edge_batch, dtype=torch.float32)
    return data, mask


class RWBaseExplainer(BaseExplainer):
    """
    A class for the Relational-PGExplainer (https://arxiv.org/abs/2011.04573).

    :param model_to_explain: graph classification model who's predictions we wish to explain.
    :param rel_emb: whether to use the relation embeddings
    :param temp: the temperture parameters dictacting how we sample our random graphs.
        - temp_start will be the first temperature at the first epoch
        - temp_end will be the final temperature at the last epoch
        - temperature will keep decreasing over the epochs, making the mask more deterministic.
    :param reg_coefs: reguaization coefficients used in the loss. The first item in the tuple restricts the size of the explainations, the second rescticts the entropy matrix mask.
    :params sample_bias: the bias we add when sampling random graphs.

    :function _create_explainer_input: utility;
    :function _sample_graph: utility; sample an explanatory subgraph.
    :function _loss: calculate the loss of the explainer during training.
    :function train: train the explainer
    :function explain: search for the subgraph which contributes most to the clasification decision of the model-to-be-explained.
    """

    def __init__(
        self,
        model_to_explain,
        size_reg=1,
        ent_reg=1,
        epochs=100,
        random_walk_loss=True,
        adj_aggr="max",
        teleport_prob=0.2,
        rw_topk_node=100,
        reg_loss_inside=1,
        reg_loss_outside=1,
        use_teleport_adj=True,
        max_power_iter=-1,
        threshold=0,
        force_include=False,
        edge_random_walk=False,
        rw_topk_edge=1000,
        convergence="node",
        invert_edges=False,
        ignore_head=False,
        spanning_tree=False,
        spanning_tree_node_ratio=0.7,
        with_path_loss=False,
        reg_path_loss=1,
        max_path_length=2,
        topk_tails=1,
        eval_mask_type="hard_node_mask",
        keep_edge_weight=False,
        joint_training=False,
        use_default_aggr=False,
        ego_network=True,
        factual=True,
        counter_factual=False,
        expl_gnn_model=False,
        model_to_evaluate=None,
    ):

        super().__init__(
            model_to_explain=model_to_explain,
            size_reg=size_reg,
            ent_reg=ent_reg,
            epochs=epochs,
            topk_tails=topk_tails,
            eval_mask_type=eval_mask_type,
            keep_edge_weight=keep_edge_weight,
            joint_training=joint_training,
            use_default_aggr=use_default_aggr,
            ego_network=ego_network,
            factual=factual,
            counter_factual=counter_factual,
            expl_gnn_model=expl_gnn_model,
            model_to_evaluate=model_to_evaluate,
        )

        # random walk loss
        self.random_walk_loss = random_walk_loss
        self.adj_aggr = adj_aggr
        self.teleport_prob = teleport_prob
        self.rw_topk_node = rw_topk_node
        self.reg_loss_inside = reg_loss_inside
        self.reg_loss_outside = reg_loss_outside
        self.use_teleport_adj = use_teleport_adj
        self.max_power_iter = max_power_iter
        self.threshold = threshold
        # whether to pad with random if no edge is found
        self.pad_zero_edges = False
        # whether to force include the central nodes
        self.force_include = force_include
        # whether to simulate a walk instead of power iteration
        self.edge_random_walk = edge_random_walk
        self.rw_topk_edge = rw_topk_edge
        # whether to wait for node or edge convergence (basically equivalent)
        self.convergence = convergence
        # Whether to invert the edge direction for the RW
        self.invert_edges = invert_edges
        # Whether to ignore the head from central RW
        self.ignore_head = ignore_head
        if self.ignore_head:
            assert self.topk_tails > 0

        # the stored node distribution
        self.node_dist = None

        # Path loss setup
        self.with_path_loss = with_path_loss
        self.max_path_length = max_path_length
        self.reg_path_loss = reg_path_loss

        # spanning tree settings
        self.spanning_tree = spanning_tree
        self.spanning_tree_node_ratio = spanning_tree_node_ratio
        self.networkx_mst = False

    def reset_node_dist(self):
        self.node_dist = None

    def get_stoch_adj_matrix(self, data, edge_index, edge_batch, mask):
        """
        Get the stoch adj matrix. This adj matrix will have the following specifications:
        dim0 (row): tgt node ids
        dim1 (col): src node ids
        dim2 (val): weight for each edge type
        This is a col-stoch adj matrix where the prob to transition to a tgt node add ups to 1

        """
        batch_size = data.subgraph_num_nodes.size(0)
        # the adj matrix will have dim0: tgt node ids, dim1: src node ids, dim2: weights for each edge type
        batch_offsets = torch.cumsum(data.subgraph_num_nodes, dim=0)
        batch_offsets -= data.subgraph_num_nodes
        offsets = batch_offsets[edge_batch]
        # give offsets so each node gets a unique id
        edge_index = edge_index + offsets
        central_node_index = data.central_node_index + batch_offsets.unsqueeze(1)

        num_total_nodes = data.subgraph_num_nodes.sum()
        # use sparse tensor to fit in memory (row: tgt, col: src)
        stoch_adj = torch.sparse_coo_tensor(
            edge_index.flip(0),
            mask,
            (num_total_nodes, num_total_nodes),
            dtype=torch.float64,
        )
        stoch_adj = stoch_adj.coalesce()
        # compute the softmax over the rows for each column s.t. the rows for each col sums upto 1
        stoch_adj = torch.sparse.softmax(stoch_adj, dim=0)

        # Teleport probability
        if self.use_teleport_adj:
            teleport_edge_index = torch.tensor(
                [[], []], device=edge_index.device, dtype=edge_index.dtype
            )
            for i in range(batch_size):
                nodes = (
                    torch.arange(data.subgraph_num_nodes[i], device=edge_index.device)
                    + batch_offsets[i]
                )
                teleport_edges = torch.cartesian_prod(nodes, central_node_index[i]).T
                teleport_edge_index = torch.cat(
                    (teleport_edge_index, teleport_edges), dim=1
                )

            prob = torch.ones(
                teleport_edge_index.size(1), device=teleport_edge_index.device
            )
            teleport_adj = torch.sparse_coo_tensor(
                teleport_edge_index.flip(0),
                prob,
                (num_total_nodes, num_total_nodes),
                dtype=torch.float64,
            )
            teleport_adj = teleport_adj.coalesce()
            teleport_adj = torch.sparse.softmax(teleport_adj, dim=0)

            # get the stochastic adjacency matrix
            stoch_adj = ((1 - self.teleport_prob) * stoch_adj) + (
                self.teleport_prob * teleport_adj
            )

        # check if the stoch adj matrix has cols summing upto 1
        sums = stoch_adj.sum(dim=0).values()
        diff = torch.abs(torch.ones_like(sums) - sums).sum()
        assert diff < 1e-3

        # starting node distribution
        # the central node can have repeated nodes for each triple
        central_nodes, counts = torch.unique(central_node_index, return_counts=True)
        prob = counts / data.central_node_index.size(1)
        node_dist = torch.zeros(
            num_total_nodes, device=prob.device, dtype=torch.float64
        )
        node_dist[central_nodes] = prob.to(node_dist.dtype)
        node_dist = node_dist.unsqueeze(1)
        assert torch.abs(node_dist.sum() - batch_size) < 1e-3

        return stoch_adj, node_dist

    def prepare_message_passing_random_walk(self, data, mask):
        """
        Prepares the data needed for message passing random walk.
        1. edge_index: edges with unique node ids in the range [0, total_num_nodes)
        2. mask: normalized mask weights
        3. real_edge_mask: mask identifying which ones are the real edges
        4. node_dist: distribution of initial nodes for the RW.
        """
        # first, give unique ids for the edge index. This ensures that the node ids will be in the range [0, total_num_nodes)
        batch_size = data.subgraph_num_nodes.size(0)
        batch_offsets = torch.cumsum(data.subgraph_num_nodes, dim=0)
        batch_offsets -= data.subgraph_num_nodes
        offsets = batch_offsets[data.edge_batch]
        # give offsets so each node gets a unique id
        if self.invert_edges:
            edge_index = data.edge_index.flip(0)
        else:
            edge_index = data.edge_index
        edge_index = edge_index + offsets
        if self.ignore_head:
            central_node_index = data.central_node_index[:, 1:]
        else:
            central_node_index = data.central_node_index
        central_node_index = central_node_index + batch_offsets.unsqueeze(1)
        real_edge_mask = torch.ones_like(data.edge_batch, dtype=torch.bool)
        num_total_nodes = data.subgraph_num_nodes.sum()

        # handle singletons by adding self loops, this ensures the RW operates correctly.
        node_ids = torch.arange(num_total_nodes, device=edge_index.device)
        singleton_mask = ~torch.isin(node_ids, edge_index)
        singleton = node_ids[singleton_mask]
        self_loops = torch.stack([singleton, singleton])
        singleton_mask = torch.ones_like(singleton)
        edge_index = torch.cat((edge_index, self_loops), dim=1)
        mask = torch.cat((mask, singleton_mask))
        real_edge_mask = torch.cat(
            (real_edge_mask, torch.zeros_like(singleton_mask, dtype=torch.bool))
        )

        # Normalize the mask such that the weights of all outgoing edges from each node sums to 1.
        mask = scatter_softmax(mask, edge_index[0])

        if self.use_teleport_adj:
            # add teleport edges
            teleport_edge_index = torch.tensor(
                [[], []], device=edge_index.device, dtype=edge_index.dtype
            )

            for i in range(batch_size):
                nodes = (
                    torch.arange(data.subgraph_num_nodes[i], device=edge_index.device)
                    + batch_offsets[i]
                )
                teleport_edges = torch.cartesian_prod(nodes, central_node_index[i]).T
                teleport_edge_index = torch.cat(
                    (teleport_edge_index, teleport_edges), dim=1
                )

            prob = torch.ones(
                teleport_edge_index.size(1), device=teleport_edge_index.device
            )
            teleport_mask = scatter_softmax(prob, teleport_edge_index[0])

            # combine the edges
            edge_index = torch.cat((edge_index, teleport_edge_index), dim=1)
            mask = (1 - self.teleport_prob) * mask
            teleport_mask = self.teleport_prob * teleport_mask
            mask = torch.cat((mask, teleport_mask))
            real_edge_mask = torch.cat(
                (real_edge_mask, torch.zeros_like(teleport_mask, dtype=torch.bool))
            )

        # sanity check, the sum should be 1
        node_sum = scatter(mask, edge_index[0])
        torch.all(torch.abs(node_sum - 1) < 1e-3)

        # starting node distribution
        # the central node can have repeated nodes for each triple
        central_nodes, counts = torch.unique(central_node_index, return_counts=True)
        prob = counts / central_node_index.size(1)
        node_dist = torch.zeros(
            num_total_nodes, device=prob.device, dtype=torch.float64
        )
        node_dist[central_nodes] = prob.to(node_dist.dtype)
        assert torch.abs(node_dist.sum() - batch_size) < 1e-3

        return edge_index, mask, real_edge_mask, node_dist

    def get_rw_topk_nodes(
        self, data, node_dist, top_num_nodes=None, return_rank=False, return_dense=False
    ):
        """
        Gettting the topk nodes from the node dist.
        By providing top_num_nodes, you can select the top num nodes to return
        Otherwise, it will assume you want to return `self.rw_topk_node`
        if return_rank, it will get the node importance rank for each subgraph in the batch
        if return_dense, the topk nodes will be returned in the dense format.
        """
        # get the topk nodes
        node_weight = torch.zeros(
            (data.subgraph_num_nodes.size(0), data.max_num_nodes),
            device=node_dist.device,
            dtype=node_dist.dtype,
        ).fill_(float("-inf"))
        indices = torch.arange(
            node_weight.size(1), device=node_weight.device
        ).expand_as(node_weight)
        valid_nodes = indices < data.subgraph_num_nodes.unsqueeze(1)
        node_weight[valid_nodes] = node_dist.squeeze()

        if top_num_nodes is None:  # get the topk nodes set by rw_topk_node
            top_num_nodes = torch.where(
                data.subgraph_num_nodes >= self.rw_topk_node,
                self.rw_topk_node,
                data.subgraph_num_nodes,
            )

        argsort = torch.argsort(node_weight, dim=1, descending=True)
        top_nodes = indices < top_num_nodes.unsqueeze(1)

        if return_dense:
            # return the sorted node id in a dense manner.
            # the padded node id will be -1
            argsort[~top_nodes] = -1
            return argsort

        s_node_id = argsort[top_nodes]  # the selected node id within each subgraph
        node_batch = torch.arange(
            node_weight.size(0), device=node_weight.device
        ).repeat_interleave(top_num_nodes)
        rank = indices[top_nodes]

        if return_rank:
            return s_node_id, node_batch, rank
        else:
            return s_node_id, node_batch

    def compute_loss_using_topk_nodes(self, data, mask, s_node_id, node_batch):
        # each edge in every subgraph gets a unique id
        offset = data.edge_batch * data.num_nodes
        unique_edges = data.edge_index + offset
        selected_nodes = s_node_id + node_batch * data.num_nodes
        edges_inside_mask = torch.all(torch.isin(unique_edges, selected_nodes), dim=0)

        if torch.any(edges_inside_mask):
            loss_inside_subgraph = -mask[edges_inside_mask].mean()
        else:
            loss_inside_subgraph = 0

        if torch.any(~edges_inside_mask):
            loss_outside_subgraph = mask[~edges_inside_mask].mean()
        else:
            loss_outside_subgraph = 0
        return loss_inside_subgraph, loss_outside_subgraph

    def pad_no_edges(self, data, num_edges, unique_edges, top_num_edges, edge_mask):
        # this will have issues when # edges is 0.
        # randomly pick edges that has any of the central node index
        batch_id = (num_edges == 0).nonzero()
        central_nodes = (
            data.central_node_index[batch_id.squeeze()] + batch_id * data.num_nodes
        )
        # the edges that has any of the central nodes
        random_mask = torch.any(
            torch.isin(unique_edges, central_nodes.flatten()), dim=0
        )
        # the edges for each batch
        batch_edge_mask = data.edge_batch.unsqueeze(0) == batch_id
        batched_edge_mask = torch.logical_and(batch_edge_mask, random_mask.unsqueeze(0))
        # get how many candidate edges each subgraph has
        # for those that has 0 candidate edges, it will randomly pick edges
        num_candidates = batched_edge_mask.sum(dim=1)
        zero_mask = num_candidates == 0
        batched_edge_mask[zero_mask] = batch_edge_mask[zero_mask]
        num_candidates = batched_edge_mask.sum(dim=1)
        #
        # shuffle the edges
        shuffle = torch.rand(batched_edge_mask.shape, device=batched_edge_mask.device)
        # get the randomly shuffled edges matching budget
        shuffle[~batched_edge_mask] = float("-inf")
        argsort = torch.argsort(shuffle, descending=True, dim=1)
        budget = top_num_edges[batch_id.squeeze(1)]
        budget = torch.min(budget, num_candidates)
        assert torch.all(budget > 0)
        indices = torch.arange(shuffle.size(1), device=batch_id.device).expand_as(
            shuffle
        )
        random_mask = indices < budget.unsqueeze(1)
        indices = argsort[random_mask]
        edge_mask[indices] = True
        return edge_mask

    def networkx_spanning_tree(
        self,
        edge_index,
        edge_batch,
        edge_type,
        mask,
        edges,
        num_nodes,
        num_edges,
        top_num_edges,
    ):
        # create networkx object
        graph_data = Data(
            edge_index=edge_index,
            mask=mask,
            num_nodes=num_nodes.sum(),
            edge_id=torch.arange(edge_index.size(1)),
            edge_batch=edge_batch,
        )
        timer1 = time.time()
        G = to_networkx(graph_data, edge_attrs=["mask", "edge_id", "edge_batch"])
        timer2 = time.time()
        logging.warning(f"Converting to Networkx took {timer2-timer1:.2f}s")
        G = convert_to_undirected(G)
        timer1 = time.time()
        logging.warning(f"Converting to Undirected took {timer1-timer2:.2f}s")
        T = nx.maximum_spanning_tree(G, weight="mask")

        spanning_num_edges = torch.zeros_like(num_edges)
        spanning_edge_id = []
        for edge in T.edges(data=True):
            edge_id = edge[-1]["edge_id"]
            batch_id = edge[-1]["edge_batch"]
            if spanning_num_edges[batch_id] < top_num_edges[batch_id]:
                spanning_num_edges[batch_id] += 1
                spanning_edge_id.append(torch.tensor(edge_id))
        timer2 = time.time()
        logging.warning(f"Getting Spanning Tree took {timer2-timer1:.2f}s")
        spanning_edge_id = torch.stack(spanning_edge_id)
        # the selected edges are the following:
        sel_mask = mask[spanning_edge_id]
        sel_edges = edges[:, spanning_edge_id]
        sel_edge_type = edge_type[spanning_edge_id]
        sel_edge_batch = edge_batch[spanning_edge_id]
        return (
            sel_mask,
            sel_edges,
            sel_edge_type,
            sel_edge_batch,
            spanning_num_edges,
            spanning_edge_id,
        )

    def get_spanning_trees(
        self, data, mask, top_num_edges, get_edge_mask, get_num_edges
    ):
        # calculate how many nodes to include
        best = torch.min(
            top_num_edges * self.spanning_tree_node_ratio, data.subgraph_num_nodes
        )
        best = torch.floor(best).to(torch.int32)
        # get the edge mask corresponding to the induced graph's edges.
        edge_mask = get_edge_mask(best, mask, self.threshold)
        # get the data of the induced graph
        mask = mask[edge_mask]
        edges = data.edge_index[:, edge_mask]
        edge_type = data.edge_type[edge_mask]
        edge_batch = data.edge_batch[edge_mask]
        # convert multidigraph to digraph
        s_data = Data(
            edge_index=edges,
            edge_type=edge_type,
            edge_batch=edge_batch,
            num_nodes=data.num_nodes,
            edge_filter=data.edge_filter,
        )
        edges, edge_batch, mask, edge_type = convert_multigraph_to_digraph(
            s_data, mask, "max", handle_singletons=False, return_edge_type=True
        )
        # create a node mask and edge mask of the induced graph
        node_mask, edge_mask = self.get_node_mask_and_edge_mask(
            data, edge_batch, edges, edge_type
        )
        # compute the num edges and num nodes in the induced graph
        num_edges = edge_mask.sum(dim=-1)
        num_nodes = node_mask.sum(dim=-1)
        node_batch, node_id = node_mask.nonzero().T
        edge_batch, edge_id = edge_mask.nonzero().T
        # nodes
        # make an offset so that each node_id will be unique
        offsets = node_batch * data.num_nodes
        # the unique_node_id for all of the nodes (sorted)
        unique_node_id = node_id + offsets
        # edges
        # make an offset so that each node in the edge will match the unique_node_id
        offsets = (edge_batch * data.num_nodes).unsqueeze(0)
        unique_edges = data.original_edge_index[:, edge_id] + offsets
        # relabel the node_id in edge_index
        assert torch.all(torch.isin(unique_edges, unique_node_id))
        # assign for every node in edges what indicies of unique_node_id it is
        edge_index = torch.bucketize(unique_edges, unique_node_id, right=False)

        if self.networkx_mst:

            (
                sel_mask,
                sel_edges,
                sel_edge_type,
                sel_edge_batch,
                spanning_num_edges,
                spanning_edge_id,
            ) = self.networkx_spanning_tree(
                edge_index,
                edge_batch,
                edge_type,
                mask,
                edges,
                num_nodes,
                num_edges,
                top_num_edges,
            )
        else:
            # first get undirected edge index based on the edge weights.
            # create adj matrix of size (total_num_nodes, total_num_nodes)
            # each item will hold the corresponding edge weight
            total_num_nodes = torch.max(edge_index) + 1
            adj = torch.zeros(
                (total_num_nodes, total_num_nodes), device=edge_index.device
            ).fill_(float("-inf"))
            adj[edge_index[0], edge_index[1]] = mask
            # another adj indicating the edge id
            adj_id = torch.zeros_like(adj).to(torch.int64)
            edge_indices = torch.arange(edge_index.size(1), device=mask.device)
            adj_id[edge_index[0], edge_index[1]] = edge_indices

            # create a mask that indicates whether the upper/lower triangle has a higher weight than the lower/higher
            # note: self loop gets ignored here.
            higher = adj > adj.T
            undir_mask = adj[higher]
            undir_edges = higher.nonzero().T
            undir_edge_id = adj_id[higher]
            # add self loops
            self_loops = edge_index[0] == edge_index[1]
            undir_mask = torch.cat((undir_mask, mask[self_loops]))
            undir_edges = torch.cat((undir_edges, edge_index[:, self_loops]), dim=1)
            undir_edge_id = torch.cat((undir_edge_id, edge_indices[self_loops]))

            sel_mask, spanning_edge_id = kruskal_mst_edges(
                undir_mask, undir_edges, undir_edge_id
            )
            sel_edges = edges[:, spanning_edge_id]
            sel_edge_type = edge_type[spanning_edge_id]
            sel_edge_batch = edge_batch[spanning_edge_id]
            b, c = torch.unique(sel_edge_batch, return_counts=True)
            spanning_num_edges = torch.zeros_like(num_edges)
            spanning_num_edges[b] = c

        # For any remaining nodes, add the edges with highest weights
        max_num_edges = max(num_edges)
        # create a tensor (batch_size, max_num_edges) that holds the edge_id
        batched_edge_id = torch.zeros(
            (len(num_edges), max_num_edges), device=num_edges.device
        ).to(torch.int32)
        indices = torch.arange(max_num_edges, device=num_edges.device).expand_as(
            batched_edge_id
        )
        indices_mask = indices < num_edges.unsqueeze(1)
        edge_indices = indices[indices_mask]
        batched_edge_id[edge_batch, edge_indices] = torch.arange(
            edge_index.size(1), dtype=torch.int32, device=num_edges.device
        )
        # create a tensor (batch_size, max_num_edges) that holds the edge weight
        batched_edge_weight = (
            torch.zeros_like(batched_edge_id).to(mask.dtype).fill_(float("-inf"))
        )
        batched_edge_weight[edge_batch, edge_indices] = mask
        # put -inf to the selected edges
        batched_edge_weight[sel_edge_batch, edge_indices[spanning_edge_id]] = float(
            "-inf"
        )
        # get the remaining edge budget
        budget = torch.min(
            num_edges - spanning_num_edges, top_num_edges - spanning_num_edges
        )
        argsort = torch.argsort(batched_edge_weight, dim=1, descending=True)
        budget_mask = indices < budget.unsqueeze(1)

        pad_edge_indices = argsort[budget_mask]
        pad_edge_batch = torch.arange(
            len(num_edges), device=num_edges.device
        ).repeat_interleave(budget)
        pad_edge_id = batched_edge_id[pad_edge_batch, pad_edge_indices]
        pad_mask = mask[pad_edge_id]
        pad_edges = edges[:, pad_edge_id]
        pad_edge_type = edge_type[pad_edge_id]

        mask = torch.cat((sel_mask, pad_mask))
        edges = torch.cat((sel_edges, pad_edges), dim=1)
        edge_type = torch.cat((sel_edge_type, pad_edge_type))
        edge_batch = torch.cat((sel_edge_batch, pad_edge_batch))

        # we have to sort it again such that it is ordered by the batches
        edge_batch, indices = torch.sort(edge_batch)
        mask = mask[indices]
        edges = edges[:, indices]
        edge_type = edge_type[indices]
        return mask, edges, edge_type, edge_batch

    def efficient_hard_edge_masks_rw(self, data, mask, node_dist, top_num_edges):
        """
        Finds the max number of nodes such that the number of edges < budget using binary search.
        """
        node_id = self.get_rw_topk_nodes(
            data, node_dist, data.subgraph_num_nodes, return_dense=True
        )
        indices = torch.arange(
            node_id.size(1), device=data.subgraph_num_nodes.device
        ).expand_as(node_id)
        # each edge in every subgraph gets a unique id
        offset = data.edge_batch * data.num_nodes
        unique_edges = data.edge_index + offset

        def get_edge_mask(rank_count, mask, threshold):
            # get the selected nodes for that rank
            selected_node_mask = indices < rank_count.unsqueeze(1)
            selected_node_id = node_id[selected_node_mask]
            node_batch = torch.arange(
                node_id.size(0), device=node_id.device
            ).repeat_interleave(rank_count)
            selected_nodes = selected_node_id + node_batch * data.num_nodes

            # get a mask to indicate the edges within the selected nodes
            edge_mask = torch.all(torch.isin(unique_edges, selected_nodes), dim=0)
            # only select those edges that satisfy the threshold
            threshold_mask = mask >= threshold
            edge_mask = torch.logical_and(edge_mask, threshold_mask)
            return edge_mask

        def get_num_edges(rank_count=None, mask=None, threshold=None, edge_mask=None):
            if edge_mask is None:
                edge_mask = get_edge_mask(rank_count, mask, threshold)
            # get the number of edges for each subgraph
            num_edges = scatter(
                edge_mask.to(torch.long),
                data.edge_batch,
                dim=0,
                dim_size=data.subgraph_num_edges.size(0),
                reduce="sum",
            )
            return num_edges

        if self.spanning_tree:
            mask, edges, edge_type, edge_batch = self.get_spanning_trees(
                data, mask, top_num_edges, get_edge_mask, get_num_edges
            )

        else:
            # stores the lower rank
            rank_low = torch.zeros_like(data.subgraph_num_nodes)
            # stores the higher rank
            rank_high = data.subgraph_num_nodes.clone()
            # stores the best rank so far
            best = torch.zeros_like(data.subgraph_num_nodes)

            while torch.any(rank_low <= rank_high):
                rank_mid = (rank_high + rank_low) // 2
                num_edges = get_num_edges(rank_mid, mask, self.threshold)

                met_quota = num_edges > top_num_edges
                # for the graphs that has too many edges, try a smaller rank
                rank_high[met_quota] = rank_mid[met_quota] - 1
                # for the graphs that has too few edges, try a larger rank
                # first store the current rank as the best
                best[~met_quota] = rank_mid[~met_quota]
                rank_low[~met_quota] = rank_mid[~met_quota] + 1

            edge_mask = get_edge_mask(best, mask, self.threshold)
            num_edges = get_num_edges(best, mask, self.threshold)

            if torch.any(num_edges == 0) and self.pad_zero_edges:
                edge_mask = self.pad_no_edges()

            mask = mask[edge_mask]
            edges = data.edge_index[:, edge_mask]
            edge_type = data.edge_type[edge_mask]
            edge_batch = data.edge_batch[edge_mask]
        # Only prune the unimportant edges if we are doing factual explanation.
        if self.factual_eval:
            data = recreate_data_object(data, edges, edge_type, edge_batch)
        elif self.counter_factual_eval:
            data = counter_factual_edge_filter(data, edge_mask=edge_mask)
        node_mask, edge_mask = self.get_node_mask_and_edge_mask(
            data, edge_batch, edges, edge_type
        )
        if not self.keep_edge_weight:  # the imp edges get a value 1
            mask = mask.fill_(1)
        # compute the num edges and num nodes in the explanatory subgraphs
        num_edges = edge_mask.sum(dim=-1)
        num_nodes = node_mask.sum(dim=-1)
        return data, mask, node_mask, edge_mask, num_edges, num_nodes

    def hard_edge_masks_rw(self, data, mask, node_dist, top_num_edges):
        """
        Deprecated: Inefficient.
        """
        node_id, node_batch, rank = self.get_rw_topk_nodes(
            data, node_dist, data.subgraph_num_nodes, return_rank=True
        )

        # each edge in every subgraph gets a unique id
        offset = data.edge_batch * data.num_nodes
        unique_edges = data.edge_index + offset
        unique_node_id = node_id + node_batch * data.num_nodes

        # iterate over the ranks until we meet the quota
        rank_count = 0
        # a final edge mask that indicates whether an edge is selected
        edges_inside_mask = torch.zeros_like(mask, dtype=torch.bool)
        # stores whether we met the quota for each subgraph
        met_quota = torch.zeros_like(data.subgraph_num_edges, dtype=torch.bool)
        while ~torch.all(met_quota):
            # get the selected nodes for that rank
            selected_nodes = unique_node_id[rank <= rank_count]
            # get a mask to indicate the edges within the selected nodes
            edge_mask = torch.all(torch.isin(unique_edges, selected_nodes), dim=0)
            # get the number of edges for each subgraph
            num_edges = scatter(
                edge_mask.to(torch.long),
                data.edge_batch,
                dim=0,
                dim_size=data.subgraph_num_edges.size(0),
                reduce="sum",
            )
            assert num_edges.sum() == edge_mask.sum()
            if torch.any(num_edges >= top_num_edges):
                # we met the quota for some of the subgraphs
                # get the ids of the subgraphs that met the quota
                quota_batch = (num_edges >= top_num_edges).nonzero().squeeze()
                met_quota[quota_batch] = True
                # get the mask for the edges of the subgraphs that surpassed the quota
                quota_mask = torch.isin(data.edge_batch, quota_batch)
                # give index of -1 to the unique edge index thus no more of the edges from that subgraph gets selected
                unique_edges[:, quota_mask] = -1
                # for the subgraphs that surpassed the quota, we will have to unselect the edges
                quota_batch = (num_edges > top_num_edges).nonzero().squeeze()
                quota_mask = torch.isin(data.edge_batch, quota_batch)
                # mark the newly selected edges as False
                edge_mask[quota_mask] = False
            # update the final mask of edges inside
            edges_inside_mask |= edge_mask
            rank_count += 1

        mask = mask[edges_inside_mask]
        edges = data.edge_index[:, edges_inside_mask]
        edge_type = data.edge_type[edges_inside_mask]
        edge_batch = data.edge_batch[edges_inside_mask]
        # Only prune the unimportant edges if we are doing factual explanation.
        if self.factual_eval:
            data = recreate_data_object(data, edges, edge_type, edge_batch)
        elif self.counter_factual_eval:
            data = counter_factual_edge_filter(data, edge_mask=edges_inside_mask)
        node_mask, edge_mask = self.get_node_mask_and_edge_mask(
            data, edge_batch, edges, edge_type
        )
        if not self.keep_edge_weight:  # the imp edges get a value 1
            mask = mask.fill_(1)
        # compute the num edges and num nodes in the explanatory subgraphs
        num_edges = edge_mask.sum(dim=-1)
        num_nodes = node_mask.sum(dim=-1)
        return data, mask, node_mask, edge_mask, num_edges, num_nodes

    def hard_node_masks_rw(self, data, mask, node_dist, top_num_nodes):
        s_node_id, node_batch = self.get_rw_topk_nodes(data, node_dist, top_num_nodes)
        # each edge in every subgraph gets a unique id
        offset = data.edge_batch * data.num_nodes
        unique_edges = data.edge_index + offset
        selected_nodes = s_node_id + node_batch * data.num_nodes
        edges_inside_mask = torch.all(torch.isin(unique_edges, selected_nodes), dim=0)
        mask = mask[edges_inside_mask]
        edges = data.edge_index[:, edges_inside_mask]
        edge_type = data.edge_type[edges_inside_mask]
        edge_batch = data.edge_batch[edges_inside_mask]
        if self.factual_eval:
            data = recreate_data_object(data, edges, edge_type, edge_batch)
        elif self.counter_factual_eval:
            data = counter_factual_edge_filter(data, edge_mask=edges_inside_mask)
        node_mask, edge_mask = self.get_node_mask_and_edge_mask(
            data, edge_batch, edges, edge_type
        )
        if not self.keep_edge_weight:  # the imp edges get a value 1
            mask = mask.fill_(1)
        # compute the num edges and num nodes in the explanatory subgraphs
        num_edges = edge_mask.sum(dim=-1)
        num_nodes = node_mask.sum(dim=-1)
        return data, mask, node_mask, edge_mask, num_edges, num_nodes

    def transform_mask_rw(self, data, mask, node_dist):
        """Gets hard mask for RW based on the node importance

        if hard_node_mask, get the topk nodes
        if hard_edge_mask, keep populating using the top ranked nodes until the quota is met.
        """
        if self.eval_mask_type == "hard_node_mask_top_ratio":
            ...

        elif self.eval_mask_type == "hard_node_mask_top_k":
            if self.hard_node_mask_top_k < 0:
                raise ValueError(
                    "Please set the correct top k for hard node mask top k"
                )
            top_num_nodes = torch.where(
                data.subgraph_num_nodes >= self.hard_node_mask_top_k,
                self.hard_node_mask_top_k,
                data.subgraph_num_nodes,
            )
            return self.hard_node_masks_rw(data, mask, node_dist, top_num_nodes)

        elif self.eval_mask_type == "hard_edge_mask_threshold":
            if self.hard_edge_mask_threshold < 0:
                raise ValueError("Please set the correct threshold for hard edge mask")
            ...

        elif self.eval_mask_type == "hard_edge_mask_top_k":
            if self.hard_edge_mask_top_k <= 0:
                raise ValueError(
                    "Please set the correct number of edges for hard edge mask"
                )
            top_num_edges = torch.where(
                data.subgraph_num_edges >= self.hard_edge_mask_top_k,
                self.hard_edge_mask_top_k,
                data.subgraph_num_edges,
            )
            # return self.hard_edge_masks_rw(data, mask, node_dist, top_num_edges)
            return self.efficient_hard_edge_masks_rw(
                data, mask, node_dist, top_num_edges
            )

        elif self.eval_mask_type == "hard_edge_mask_top_ratio":
            if self.hard_edge_mask_top_ratio <= 0:
                raise ValueError("Please set the correct ratio for hard edge mask")
            top_num_edges = (
                data.subgraph_num_edges * self.hard_edge_mask_top_ratio
            ).to(torch.int32)
            return self.efficient_hard_edge_masks_rw(
                data, mask, node_dist, top_num_edges
            )

        else:
            raise NotImplementedError

    def message_passing_random_walk(self, data, mask):
        """
        This function performs random walk via message passing instead of power iteration.
        The benefit of this function over the power-iteration style is that it does not require the construction
        of stochastic adjacency matrix. This implies that we can keep the data structure of the graph as a multi-directed graph
        instead of converting the graph to a directed graph. This can also bypass a lot of annoying index logging that needs to happen
        by using sparse matrix representation and coalesce().
        """

        # test with a small graph to see if it is working
        # data, rw_mask = small_example(data, mask)
        edge_index, rw_mask, real_edge_mask, node_dist = (
            self.prepare_message_passing_random_walk(data, mask)
        )

        # perform message passing random walk
        if self.max_power_iter > 0:
            max_power_iter = self.max_power_iter
        else:
            max_power_iter = float("inf")

        eps = 1e-3
        diff = float("inf")
        initial_dist = node_dist.clone()
        edge_dist = None
        edge_diff = float("inf")
        count = 0
        while diff > eps and count < max_power_iter:
            count += 1
            src = edge_index[0]
            tgt = edge_index[1]
            node_w = node_dist[src]
            edge_w = rw_mask * node_w

            # inspect the message
            if edge_dist is not None:
                edge_diff = torch.abs(edge_dist - edge_w).sum()
            edge_dist = edge_w
            if self.use_teleport_adj:
                new_node_dist = scatter(edge_w, tgt)
            else:
                new_node_dist = self.teleport_prob * initial_dist + (
                    1 - self.teleport_prob
                ) * scatter(edge_w, tgt)
            node_diff = torch.abs(new_node_dist - node_dist).sum()
            node_dist = new_node_dist
            if self.convergence == "node":
                diff = node_diff
            elif self.convergence == "edge":
                diff = edge_diff
            else:
                raise ValueError("Convergence has to be either node or edge.")

        batch_size = data.subgraph_num_nodes.size(0)
        assert torch.abs(node_dist.sum() - batch_size) < eps
        assert torch.abs(edge_dist.sum() - batch_size) < eps

        if self.edge_random_walk:
            # get the mask only for the real edges
            rw_mask = edge_dist[real_edge_mask]
            return rw_mask
        else:
            raise NotImplementedError

    def get_top_edge_mask(self, data, mask, k):
        # select the topk edges
        # clip so that max num edges one can return is the num edges in the subgraph
        top_num_edges = torch.where(
            data.subgraph_num_edges >= k,
            k,
            data.subgraph_num_edges,
        )
        # edge_weight will hold the mask
        edge_weight = (
            torch.zeros(data.edge_filter.shape, device=mask.device)
            .fill_(float("-inf"))
            .to(mask.dtype)
        )
        # find out which edges are valid
        indices = torch.arange(
            edge_weight.size(1), device=edge_weight.device
        ).expand_as(edge_weight)
        valid_edges = indices < data.subgraph_num_edges.unsqueeze(1)
        # fill in edge_weight with mask
        edge_weight[valid_edges] = mask
        # sort the edge_weight based on mask
        argsort = torch.argsort(edge_weight, dim=1, descending=True)
        # get the edge_id of the edges with the highest mask value
        top_edges = indices < top_num_edges.unsqueeze(1)
        edge_id = argsort[top_edges]
        batch_id = torch.arange(
            edge_weight.size(0), device=edge_weight.device
        ).repeat_interleave(top_num_edges)
        # the edge_id needs to be updated according to the batch_id
        batch_offset = data.subgraph_num_edges.cumsum(0) - data.subgraph_num_edges
        edge_id = edge_id + batch_offset[batch_id]
        # create a mask indicating True for any edges that is part of the highest mask value
        top_edge_mask = torch.zeros_like(mask, dtype=torch.bool)
        top_edge_mask[edge_id] = True
        return top_edge_mask

    def do_edge_random_walk(self, data, mask, return_loss=True):
        """Compute the random walk loss
        Uses sparse tensor multiplication

        return_loss: whether to return the loss
        store_walk: whether to store the walk result
        """
        rw_mask = self.message_passing_random_walk(data, mask)
        # convert rw_mask to log for better numerical stability.
        rw_mask = torch.log(rw_mask + 1e-12).to(torch.float32)

        if return_loss:
            top_edge_mask = self.get_top_edge_mask(data, rw_mask, self.rw_topk_edge)
            if torch.any(top_edge_mask):
                loss_inside_subgraph = -mask[top_edge_mask].mean()
            else:
                loss_inside_subgraph = 0

            if torch.any(~top_edge_mask):
                loss_outside_subgraph = mask[~top_edge_mask].mean()
            else:
                loss_outside_subgraph = 0
            # get the loss
            rw_loss = (
                self.reg_loss_inside * loss_inside_subgraph
                + self.reg_loss_outside * loss_outside_subgraph
            )
            return rw_loss
        else:
            assert not self.keep_edge_weight, "Mask value converted by Log."
            assert "hard" in self.eval_mask_type, "Mask value converted by Log."
            return rw_mask

    def do_random_walk(self, data, mask, return_loss=True, store_walk=False):
        """Compute the random walk loss
        Uses sparse tensor multiplication

        return_loss: whether to return the loss
        store_walk: whether to store the walk result
        """
        if store_walk and self.node_dist is not None:
            # get the masks based on the hard masks using the stored node distribution
            return self.transform_mask_rw(data, mask, self.node_dist)

        # KG is inherently a multi-di-graph, have to create one adjacency matrix based on some sort of aggregation
        # if any of the central nodes are a singleton, add self loops for these nodes so that the stoch. adj. matrix can work.
        # edge_index, edge_batch, rw_mask = self.prepare_edges_for_rw(data, mask)
        edge_index, edge_batch, rw_mask = convert_multigraph_to_digraph(
            data, mask, self.adj_aggr, handle_singletons=True
        )

        if self.with_path_loss:
            path_loss = get_path_loss(
                data, edge_index, edge_batch, rw_mask, self.max_path_length
            )

        # get the stochastic adj matrix and the starting node dist
        stoch_adj, node_dist = self.get_stoch_adj_matrix(
            data, edge_index, edge_batch, rw_mask
        )

        # power method
        if self.max_power_iter > 0:
            max_power_iter = self.max_power_iter
        else:
            max_power_iter = float("inf")

        eps = 1e-3
        diff = float("inf")
        initial_dist = node_dist.clone()
        count = 0
        while diff > eps and count < max_power_iter:
            count += 1
            if self.use_teleport_adj:
                new_node_dist = torch.sparse.mm(stoch_adj, node_dist)
            else:
                new_node_dist = self.teleport_prob * initial_dist + (
                    1 - self.teleport_prob
                ) * torch.sparse.mm(stoch_adj, node_dist)
            diff = torch.abs(new_node_dist - node_dist)
            diff = diff.sum()
            node_dist = new_node_dist

        batch_size = data.subgraph_num_nodes.size(0)
        assert torch.abs(node_dist.sum() - batch_size) < eps

        if self.force_include:
            central_node_mask = initial_dist > 0
            # include the central nodes.
            node_dist[central_node_mask] = float("inf")

        if store_walk and self.node_dist is None:
            # store the node distribution for future evaluations
            self.node_dist = node_dist
            return

        if return_loss:
            # select the topk nodes
            s_node_id, node_batch = self.get_rw_topk_nodes(data, node_dist)

            # get the loss
            loss_inside_subgraph, loss_outside_subgraph = (
                self.compute_loss_using_topk_nodes(data, mask, s_node_id, node_batch)
            )
            rw_loss = (
                self.reg_loss_inside * loss_inside_subgraph
                + self.reg_loss_outside * loss_outside_subgraph
            )
            if self.with_path_loss:
                path_loss = self.reg_path_loss * path_loss
            else:
                path_loss = torch.tensor(0, device=mask.device)

            return rw_loss, path_loss
        else:
            # get the masks based on the hard masks
            return self.transform_mask_rw(data, mask, node_dist)

    ''' * Deprecated, using data_util.convert_multigraph_to_digraph now *
    def prepare_edges_for_rw(self, data, mask):
        """
        1. Knowledge graph can be multigraph, convert this to digraph (only one edge in u, v) by aggregating the weights for each relation.
        2. The central nodes / nodes in masked edges can be a singleton. Add self loop so the power method works.
        Aggr options: max, sum, mean
        """
        # each edge in every subgraph gets a unique id
        offset = data.edge_batch * data.num_nodes
        unique_edges = data.edge_index + offset

        # for each original edge, inverse_indices is the corresponding index in unique edges
        unique_edges, inverse_indices = torch.unique(
            unique_edges, return_inverse=True, dim=1
        )

        num_relations = torch.max(data.edge_type) + 1
        # for each unique edge, multi_edge_weights will get the score of the mask for each available edge type
        multi_edge_weights = torch.zeros(
            (unique_edges.size(1), num_relations), device=unique_edges.device
        )
        multi_edge_weights[inverse_indices, data.edge_type] = mask
        if self.adj_aggr == "max":
            # for each unique edge, take the relation with highest weight (lowest negative weight)
            mask, _ = torch.max(multi_edge_weights, dim=-1)
        elif self.adj_aggr == "mean":
            dense_edge_filter = torch.zeros_like(multi_edge_weights)
            dense_edge_filter[inverse_indices, data.edge_type] = 1
            norm = dense_edge_filter.sum(dim=-1)
            mask = torch.sum(multi_edge_weights, dim=-1) / norm
        elif self.adj_aggr == "sum":
            mask = torch.sum(multi_edge_weights, dim=-1)
        else:
            raise ValueError(f"Unknown adj_aggr type: {self.adj_aggr}")

        batch_size = data.edge_filter.size(0)
        # check if there are any singleton nodes
        # if so, add self loops so it doesn't mess up the random walk
        indices = (
            torch.arange(data.max_num_nodes, device=data.edge_index.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        valid_nodes = indices < data.subgraph_num_nodes.unsqueeze(1)
        node_ids = indices[valid_nodes]
        node_ids = node_ids + data.node_batch * data.num_nodes
        singleton_mask = ~torch.isin(node_ids, unique_edges)
        singleton = node_ids[singleton_mask]
        singleton_batch = data.node_batch[singleton_mask]
        self_loops = torch.stack([singleton, singleton])
        singleton_mask = torch.ones_like(singleton_batch)
        mask = torch.cat((mask, singleton_mask))

        # map the unique edges back to its original
        # get the edge_batch for the unique edges
        unique_edge_batch = torch.zeros(
            (unique_edges.size(1), batch_size),
            device=unique_edges.device,
            dtype=torch.long,
        )
        unique_edge_batch[inverse_indices, data.edge_batch] = 1
        assert torch.all(
            unique_edge_batch.sum(dim=-1) == 1
        )  # there should only be one batch id for each unique edge
        edge_batch = unique_edge_batch.nonzero().T[1]
        edge_batch = torch.cat((edge_batch, singleton_batch))
        rev_offset = edge_batch * data.num_nodes
        unique_edges = torch.cat((unique_edges, self_loops), dim=1)

        edge_index = unique_edges - rev_offset

        return edge_index, edge_batch, mask
    '''
