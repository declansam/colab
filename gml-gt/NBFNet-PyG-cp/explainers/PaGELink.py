import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import degree
import copy
import time
from collections import defaultdict
from torch import optim
from explainers.BaseExplainer import BaseExplainer
from tqdm import tqdm
from nbfnet import customized_func
from explainers.data_util import (
    remove_edges,
    recreate_data_object,
    counter_factual_edge_filter,
)
from explainers.path_utils import *
from tqdm import tqdm


def remove_edges_of_high_degree_nodes(data, max_degree=10):
    """
    For all the nodes with degree higher than `max_degree`,
    except nodes in `data.central_node_index`, remove their edges.

    Parameters
    ----------
    data : torch geometric graph

    max_degree : int

    Returns
    -------
    data : torch geometric graph
        Pruned graph with edges of high degree nodes removed

    """
    degrees_in = customized_func.customized_degree(
        data.batched_edge_index[1],
        preserved_edges=data.edge_filter,
        dim=-1,
        num_nodes=data.max_num_nodes,
    )
    if degrees_in.ndim == 1:
        degrees_in = degrees_in.unsqueeze(0)
    high_degree_mask = degrees_in > max_degree
    # preserve nodes in data.central_node_index
    batch_id = torch.arange(data.edge_filter.size(0)).repeat_interleave(
        data.central_node_index.size(1)
    )
    high_degree_mask[batch_id, data.central_node_index.flatten()] = False

    # remove edges of high degree nodes
    node_batch, node_id = high_degree_mask.nonzero().T
    offsets = (
        node_batch * data.num_nodes
    )  # make an offset so that each node_id will be unique
    unique_node_id = node_id + offsets
    offsets = (data.edge_batch * data.num_nodes).unsqueeze(0)
    unique_edges = data.edge_index + offsets
    edge_mask = torch.any(torch.isin(unique_edges, unique_node_id), dim=0)
    data = remove_edges(data, edge_mask)

    return data


def remove_edges_except_k_core_graph(data, k):
    """
    Find the `k`-core of `data`.
    Only isolate the low degree nodes by removing theirs edges
    except nodes in `data.central_node_index`, remove their edges

    Parameters
    ----------
    data : torch geometric graph
    k : int

    Returns
    -------
    data : torch geometric graph
        The k-core graph
    """
    degrees_in = customized_func.customized_degree(
        data.batched_edge_index[1],
        preserved_edges=data.edge_filter,
        dim=-1,
        num_nodes=data.max_num_nodes,
    )
    if degrees_in.ndim == 1:
        degrees_in = degrees_in.unsqueeze(0)
    k_core_mask = (degrees_in > 0) & (degrees_in < k)
    # preserve nodes in data.central_node_index
    batch_id = torch.arange(data.edge_filter.size(0)).repeat_interleave(
        data.central_node_index.size(1)
    )
    k_core_mask[batch_id, data.central_node_index.flatten()] = False

    while k_core_mask.any():
        node_batch, node_id = k_core_mask.nonzero().T
        offsets = (
            node_batch * data.num_nodes
        )  # make an offset so that each node_id will be unique
        unique_node_id = node_id + offsets
        offsets = (data.edge_batch * data.num_nodes).unsqueeze(0)
        unique_edges = data.edge_index + offsets
        edge_mask = torch.any(torch.isin(unique_edges, unique_node_id), dim=0)
        data = remove_edges(data, edge_mask)
        degrees_in = customized_func.customized_degree(
            data.batched_edge_index[1],
            preserved_edges=data.edge_filter,
            dim=-1,
            num_nodes=data.max_num_nodes,
        )
        if degrees_in.ndim == 1:
            degrees_in = degrees_in.unsqueeze(0)
        k_core_mask = (degrees_in > 0) & (degrees_in < k)
        k_core_mask[batch_id, data.central_node_index.flatten()] = False

    return data


def get_neg_path_score(data, mask):
    """
    Compute the negative path score for the shortest path algorithm.
    Gives a degree score of 0 to nodes in central_node_index when computing path score, so they will
    likely be included
    """
    log_mask = torch.log(mask)
    degrees_in = customized_func.customized_degree(
        data.batched_edge_index[1],
        preserved_edges=data.edge_filter,
        dim=-1,
        num_nodes=data.max_num_nodes,
    )
    if degrees_in.ndim == 1:
        degrees_in = degrees_in.unsqueeze(0)
    degrees_in = torch.log(degrees_in)
    batch_id = torch.arange(data.edge_filter.size(0)).repeat_interleave(
        data.central_node_index.size(1)
    )
    degrees_in[batch_id, data.central_node_index.flatten()] = 0
    tgt_node = data.edge_index[1]
    degrees_in = degrees_in[data.edge_batch, tgt_node]
    neg_path_score = degrees_in - log_mask
    return neg_path_score


class PaGELink(BaseExplainer):
    """
    A class for the Relational-PGExplainer (https://arxiv.org/abs/2011.04573).

    :param model_to_explain: graph classification model who's predictions we wish to explain.
    :param size_reg: Restricts the size of the explainations.
    :param ent_reg: Rescticts the entropy matrix mask.
    :param epochs: Number of epochs to train the GNNExplainer
    :param prune_graph: If true apply the max_degree and/or k-core pruning. For ablation. Default True.
    alpha : float, optional
        A higher value will make edges on high-quality paths to have higher weights
    beta : float, optional
        A higher value will make edges off high-quality paths to have lower weights
    log : bool, optional
        If True, it will log the computation process, default to True.
    :param eval_mask_type: The mask type used for evaluation
    :param return_detailed_loss: whether to return detailed loss
    :param joint_training: whether to conduct joint training of the GNN.

    """

    def __init__(
        self,
        model_to_explain,
        lr=0.003,
        optimizer="Adam",
        epochs=100,
        prune_graph=True,
        prune_max_degree=20,
        k_core=2,
        with_path_loss=True,
        num_paths=5,
        timeout_duration=3,
        alpha=1.0,
        beta=1.0,
        topk_tails=1,
        eval_mask_type="hard_node_mask",
        keep_edge_weight=False,
        joint_training=False,
        use_default_aggr=False,
        ego_network=True,
        factual=True,
        counter_factual=False,
        model_to_evaluate=None,
    ):

        super().__init__(
            model_to_explain,
            epochs=epochs,
            topk_tails=topk_tails,
            eval_mask_type=eval_mask_type,
            keep_edge_weight=keep_edge_weight,
            joint_training=joint_training,
            use_default_aggr=use_default_aggr,
            ego_network=ego_network,
            factual=factual,
            counter_factual=counter_factual,
            model_to_evaluate=model_to_evaluate,
        )

        self.lr = lr
        self.optimizer = optimizer
        self.prune_graph = prune_graph
        self.prune_max_degree = prune_max_degree
        self.k_core = k_core
        self.with_path_loss = with_path_loss
        self.num_paths = num_paths  # num paths used in training
        self.timeout_duration = (
            timeout_duration  # max time to explore new path in evaluation
        )
        self.alpha = alpha
        self.beta = beta

        self.all_loss = defaultdict(list)
        # Too Expensive to compute shortest paths across different examples.
        assert self.topk_tails == 1
        assert self.factual + self.counter_factual == 1

    def _set_masks(self, data):
        """
        Inject the explanation mask into the message passing modules.
        :param x: features
        :param edge_index: graph representation
        """
        std = torch.nn.init.calculate_gain("relu") * torch.sqrt(
            2.0 / (2 * data.subgraph_num_nodes)
        )
        self.edge_mask = torch.nn.Parameter(
            torch.randn(data.edge_index.size(1), device=std.device)
            * std[data.edge_batch]
        )

    def _clear_masks(self):
        """
        Cleans the injected edge mask from the message passing modules. Has to be called before any new sample can be explained.
        """
        self.edge_mask = None

    def _prune_graph(self, data):
        if self.prune_max_degree > 0:
            data = remove_edges_of_high_degree_nodes(data, self.prune_max_degree)
        data = remove_edges_except_k_core_graph(data, self.k_core)
        return data

    def path_loss(self, data, mask):
        """Compute the path loss."""
        neg_path_score = get_neg_path_score(data, mask)
        path_data, path_mask, neg_path_score, unselected_edge_id = multigraph_to_graph(
            data, mask, neg_path_score
        )
        edges_in_path = sequential_path_finder(
            path_data,
            weight=neg_path_score,
            k=self.num_paths,
            timeout_duration=self.timeout_duration,
        )

        if torch.any(edges_in_path):
            loss_on_path = -path_mask[edges_in_path].mean()
        else:
            loss_on_path = 0

        offpath_mask = torch.cat((path_mask[~edges_in_path], mask[unselected_edge_id]))
        assert path_mask[edges_in_path].size(0) + offpath_mask.size(0) == mask.size(0)

        if offpath_mask.size(0) > 0:
            loss_off_path = offpath_mask.mean()
        else:
            loss_off_path = 0

        self.all_loss["loss_on_path"] += [float(loss_on_path)]
        self.all_loss["loss_off_path"] += [float(loss_off_path)]

        loss = self.alpha * loss_on_path + self.beta * loss_off_path

        return loss

    @torch.no_grad()
    def get_paths(self, data):
        """A postprocessing step that turns the mask into actual paths.
        This will continue exploring the paths until the max_budget (either num nodes or num edges)
        is met. In some cases, the loop will take a long time since there are many paths to explore.
        To counter this, we can set a timeout duration for each (src, tgt) pair.
        """
        # Retrieve final explanation
        self.eval()
        mask = torch.sigmoid(self.edge_mask)
        neg_path_score = get_neg_path_score(data, mask)
        path_data, path_mask, neg_path_score, unselected_edge_id = multigraph_to_graph(
            data, mask, neg_path_score
        )
        path_collection = sequential_path_finder_until_budget(
            path_data,
            weight=neg_path_score,
            return_paths=True,
            eval_mask_type=self.eval_mask_type,
            max_budget=getattr(self, self.eval_mask_type),
            timeout_duration=self.timeout_duration,
        )

        self.path_collection = path_collection
        self.path_data = path_data
        self.path_mask = path_mask

    def train_mask(self, data, batch):
        self._clear_masks()
        # If counter factual explanation, we don't prune it.
        if self.prune_graph and not self.counter_factual:
            data = self._prune_graph(data)
        self._set_masks(data)
        optimizer = getattr(optim, self.optimizer)([self.edge_mask], lr=self.lr)

        # Start training loop
        eweight_norm = 0
        EPS = 1e-3
        for e in range(0, self.epochs):
            optimizer.zero_grad()
            mask = torch.sigmoid(self.edge_mask)
            edge_weights = self.get_edge_weight(data, mask)

            if self.factual:
                edge_weight = edge_weights["factual"]
            else:
                edge_weight = edge_weights["counter_factual"]

            if not self.use_default_aggr:  # aggregation with regards to the weights
                data.edge_filter = edge_weight  # the edge filter used to calculate the degree info is the edge weight

            masked_pred = self.pred_model(data, batch, edge_weight=edge_weight)
            assert not torch.any(torch.isnan(masked_pred))

            if self.factual:  # factual explanation
                label = torch.ones_like(masked_pred)
            else:
                label = torch.zeros_like(masked_pred)  # counter factual

            loss = F.binary_cross_entropy_with_logits(
                masked_pred, label, reduction="mean"
            )
            self.all_loss["pred_loss"] += [loss.item()]

            # Check for early stop
            curr_eweight_norm = mask.norm()
            if abs(eweight_norm - curr_eweight_norm) < EPS:
                break
            eweight_norm = curr_eweight_norm

            # Update with path loss
            if self.with_path_loss:
                path_loss = self.path_loss(data, mask)
            else:
                path_loss = 0

            loss = loss + path_loss
            self.all_loss["total_loss"] += [loss.item()]

            loss.backward()
            optimizer.step()

        # after training, compute the explanation paths!
        if "hard" in self.eval_mask_type:
            self.get_paths(data)

    @torch.no_grad()
    def evaluate_mask(self, data, batch):
        # evaluate the performance given the computed explanation paths!
        if self.eval_mask_type == "soft_edge_mask":
            # special pipeline as it cannot use the precomputed paths as explanations
            if self.prune_graph and not self.counter_factual:
                data = self._prune_graph(data)
            mask = torch.sigmoid(self.edge_mask)
            data, mask, node_mask, edge_mask, num_edges, num_nodes = (
                self.transform_mask(data, mask)
            )
        else:
            # get the explanation based on the computed paths
            data, mask, node_mask, edge_mask, num_edges, num_nodes = (
                self.transform_mask(
                    self.path_data, self.path_mask, self.path_collection
                )
            )
        # based on the mask, get the edge weight
        edge_weight = self.get_edge_weight_eval(data, mask)

        masked_pred = self.eval_model(data, batch, edge_weight=edge_weight)
        assert not torch.any(torch.isnan(masked_pred))

        return masked_pred, node_mask, edge_mask, num_edges, num_nodes

    @torch.no_grad()
    def transform_mask(self, data, mask, path_collection=None):
        """
        Overwriting mask transformation
        """
        data = copy.copy(data)
        if "hard" in self.eval_mask_type:
            if getattr(self, f"{self.eval_mask_type}") <= 0:
                raise ValueError("Please set the corrent ratio for hard mask")

            batch_size = data.edge_filter.size(0)
            selected_edge_ids = torch.tensor(
                [], device=data.edge_index.device, dtype=torch.long
            )
            for i in range(batch_size):
                satisfied_budget = False
                paths = path_collection[i]
                selected_nodes = set()
                selected_edges = set()
                for path in paths:
                    nodes = set(path[0])
                    edge_ids = set(path[1])
                    node_union = selected_nodes.union(nodes)
                    edge_union = selected_edges.union(edge_ids)
                    if self.eval_mask_type == "hard_node_mask_top_ratio":
                        ...

                    elif self.eval_mask_type == "hard_node_mask_top_k":
                        ...

                    elif self.eval_mask_type == "hard_edge_mask_threshold":
                        ...

                    elif self.eval_mask_type == "hard_edge_mask_top_k":
                        if len(edge_union) > self.hard_edge_mask_top_k:
                            satisfied_budget = True
                            break

                    elif self.eval_mask_type == "hard_edge_mask_top_ratio":
                        ...

                    selected_nodes = node_union
                    selected_edges = edge_union

                selected_edge_ids = torch.cat(
                    [
                        selected_edge_ids,
                        torch.tensor(
                            list(selected_edges),
                            device=data.edge_index.device,
                            dtype=torch.long,
                        ),
                    ]
                )

            selected_edge_mask = torch.zeros_like(data.edge_batch, dtype=torch.bool)
            selected_edge_mask[selected_edge_ids] = True

            mask = mask[selected_edge_mask]
            edges = data.edge_index[:, selected_edge_mask]
            edge_type = data.edge_type[selected_edge_mask]
            edge_batch = data.edge_batch[selected_edge_mask]
            if self.factual_eval:
                data = recreate_data_object(data, edges, edge_type, edge_batch)
            elif self.counter_factual_eval:
                data = counter_factual_edge_filter(data, edge_mask=selected_edge_mask)

            node_mask, edge_mask = self.get_node_mask_and_edge_mask(
                data, edge_batch, edges, edge_type
            )
            if not self.keep_edge_weight:  # the imp edges get a value 1
                mask = mask.fill_(1)
            # compute the num edges and num nodes in the explanatory subgraphs
            num_edges = edge_mask.sum(dim=-1)
            num_nodes = node_mask.sum(dim=-1)

            # the location of the mask changes
            return (
                data,
                mask,
                node_mask,
                edge_mask,
                num_edges,
                num_nodes,
            )

        elif self.eval_mask_type == "soft_edge_mask":
            node_mask, edge_mask = self.get_node_mask_and_edge_mask(
                data, data.edge_batch, data.edge_index, data.edge_type
            )
            return (
                data,
                mask,
                node_mask,
                edge_mask,
                data.subgraph_num_edges,
                data.subgraph_num_nodes,
            )

        else:
            raise NotImplementedError
