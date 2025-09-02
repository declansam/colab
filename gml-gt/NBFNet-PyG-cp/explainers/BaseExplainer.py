import copy
import torch
from torch import nn
from nbfnet import tasks
from torch_geometric.utils import degree
from nbfnet import customized_func
from explainers.data_util import recreate_data_object, counter_factual_edge_filter


class CustomDDPWrapper(nn.parallel.DistributedDataParallel):
    """This wrapper is used for the parallel model to access certain methods of the explainer"""

    def set_train(self):
        self.module.set_train()


class BaseExplainer(nn.Module):
    def __init__(
        self,
        model_to_explain,
        size_reg=1,
        ent_reg=1,
        epochs=100,
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
        super().__init__()

        self.epochs = epochs
        # for the loss
        self.size_reg = size_reg
        self.ent_reg = ent_reg

        # how many topk tails of the original prediction should we consider for graph sampling
        self.topk_tails = topk_tails

        # whether to evaluate only the edges in the ego network
        self.ego_network = ego_network

        # TRUE -> Normal Aggregation, False -> aggregation w.r.t. to the weights
        self.use_default_aggr = use_default_aggr
        assert self.use_default_aggr == False

        # whether to do factual / counter factual explanation
        self.factual = factual
        self.counter_factual = counter_factual
        assert (
            factual or counter_factual
        ), "Please select factual, counter factual or both"
        # evaluation
        self.factual_eval = False
        self.counter_factual_eval = False

        # *** evaluation mask settings *** #
        self.eval_mask_type = eval_mask_type
        self.max_num_nodes_in_mask = -1
        self.hard_edge_mask_threshold = -1
        self.hard_edge_mask_top_k = -1
        self.hard_edge_mask_top_ratio = -1
        self.keep_edge_weight = keep_edge_weight

        # *** model to explain ***
        # whether to have a separate expl GNN model
        self.expl_gnn_model = expl_gnn_model
        # whether to update the pred model given the masked subgraph
        self.joint_training = joint_training
        # pred model given the masked subgraph
        self.pred_model = model_to_explain

        if self.expl_gnn_model:
            # expl model that updates the embeddings
            self.expl_model = copy.deepcopy(model_to_explain)
            # the mlp used for decoding of NBFNet will not be updated since we are only getting the embeddings.
            assert "NBFNet" in self.expl_model.__class__.__name__
            for param in self.expl_model.mlp.parameters():
                param.requires_grad = False
        else:
            self.expl_model = self.pred_model

        # whether to optimize the pred model given the masked subgraph
        if self.joint_training:
            # original model used to obtain the original pred
            self.original_model = copy.deepcopy(model_to_explain)
        else:
            self.original_model = self.pred_model

        # eval model given the expl subgraph, evaluates the quality of explanation
        if model_to_evaluate:  # if we are provided with the model used to evaluate
            self.eval_model = model_to_evaluate
        else:
            self.eval_model = self.pred_model

        if model_to_explain is not None:
            self.hops = model_to_explain.hops
            # don't update the params of the predictor if not joint training
            if not self.joint_training:
                for param in self.pred_model.parameters():
                    param.requires_grad = False

        if model_to_evaluate is not None:
            # don't update the params of the evaluator if provided
            for param in self.eval_model.parameters():
                param.requires_grad = False

        # Used to efficiently run inference for global explainers s.t.
        # it doesn't re-compute the mask for the same batch.
        self.prev_batch = None
        self.prev_mask = None

    def set_train(self):
        self.train()
        self.original_model.eval()  # the original model is always set to evaluation mode
        # if not joint training, put the pred model and eval model to eval mode
        if not self.joint_training:
            self.pred_model.eval()
            self.eval_model.eval()

    def forward(self):
        raise NotImplementedError

    @torch.no_grad()
    def full_prediction(self, data, batch):
        # the original pred is given by the fixed original_mode, and in the case we have a separate expl gnn model, the embeddings are given by the expl_model.
        original_pred, embeds, R = self.original_model(data, batch, return_emb=True)
        return original_pred, embeds, R

    def regularization_loss(self, mask):
        """
        Returns the loss score based on the given mask.
        :param masked_pred: Prediction based on the current explanation
        :param label: The label for the prediction
        :param edge_mask: Current explanaiton
        :return: loss
        """
        EPS = 1e-15

        # Based on the Loss, we should consider size_loss and mask_ent_loss applied on the node weights.

        # Regularization losses
        # size_loss = torch.sum(mask) * size_reg # SHOULDN'T THIS BE THE MEAN?
        size_loss = torch.mean(mask) * self.size_reg
        # the term mask * torch.log(mask) can lead to nan
        mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(
            1 - mask + EPS
        )
        mask_ent_loss = self.ent_reg * torch.mean(mask_ent_reg)

        return size_loss, mask_ent_loss

    def get_edge_weight(self, data, mask):
        """
        Based on the mask, gets the edge weight, depending on counter-factual or not.
        """
        edge_weights = {}

        if self.factual:  # Factual Explanation
            edge_weight = torch.zeros_like(data.edge_filter)
            edge_weight[data.edge_filter.to(torch.bool)] = mask
            edge_weights["factual"] = edge_weight

        if self.counter_factual:  # Counter-factual Explanation
            edge_weight = torch.zeros_like(data.edge_filter)
            edge_weight[data.edge_filter.to(torch.bool)] = 1 - mask
            edge_weights["counter_factual"] = edge_weight

        return edge_weights

    def get_edge_weight_eval(self, data, mask):
        edge_weight = torch.zeros_like(data.edge_filter)
        assert self.factual_eval + self.counter_factual_eval == 1
        if self.factual_eval:  # Factual Explanation
            edge_weight[data.edge_filter.to(torch.bool)] = mask  # factual edges
        if self.counter_factual_eval:
            edge_weight[data.edge_filter.to(torch.bool)] = 1  # the unselected edges
            # counter factual edges
            edge_weight[data.counter_factual_filter.to(torch.bool)] = 1 - mask
        return edge_weight

    def get_node_mask_and_edge_mask(self, data, batch_id, edges, edge_type):
        """
        Construct the node mask (batch_size, num_nodes_full_graph) indicating whether each node is included in explanatory graph
        Also, construct the node mask (batch_size, num_edges_full_graph) indicating whether each edge is included in explanatory graph
        data: the graph data
        batch_id: the batch index for each edge
        edges: the selected edges (indices in subgraph node id)
        edge_type: the edge type of selected edges
        """
        device = data.edge_filter.device
        batch_size = data.edge_filter.size(0)
        # construct the node mask (batch_size, num_nodes_full_graph) indicating whether each node is included in explanatory graph
        # first, create a tensor (batch_size, max_num_nodes) that maps from the subgraph node index to full node index
        node_mapping = torch.zeros(
            (batch_size, data.max_num_nodes), device=device, dtype=data.node_id.dtype
        )
        indices = torch.arange(data.max_num_nodes, device=device).repeat(batch_size, 1)
        node_filter = indices < data.subgraph_num_nodes.unsqueeze(1)
        node_mapping[node_filter] = data.node_id
        # then, from the s_node_id of the edges in the explanatory graph, get the corresponding full graph node_id
        # mark these nodes as True on the node mask
        node_mask = torch.zeros(
            (batch_size, data.num_nodes), device=device, dtype=torch.bool
        )
        src = edges[0]
        src = node_mapping[batch_id, src]
        node_mask[batch_id, src] = True
        tgt = edges[1]
        tgt = node_mapping[batch_id, tgt]
        node_mask[batch_id, tgt] = True

        # construct the edge mask (batch_size, num_edges_full_graph) indicating whether each edge is included in explanatory graph
        edge_mask = torch.zeros(
            (batch_size, data.original_edge_index.size(1)),
            device=device,
            dtype=torch.bool,
        )
        # find the corresponding edge id for each explanatory edge from the original edge index
        expl_edge_index = torch.stack([src, tgt])
        edge_index = torch.cat(
            [data.original_edge_index, data.original_edge_type.unsqueeze(0)]
        )
        expl_edge_index = torch.cat([expl_edge_index, edge_type.unsqueeze(0)])
        if expl_edge_index.size(1) != 0:
            edge_id, num_match = tasks.edge_match(edge_index, expl_edge_index)
            assert torch.all(num_match == 1)
            # mark these edges as True on the edge mask
            edge_mask[batch_id, edge_id] = True

        return node_mask, edge_mask

    def select_topk_edges(self, data, mask, top_num_edges):
        """get the topk edges for each triple"""
        edge_weight = torch.zeros(data.edge_filter.shape, device=mask.device).fill_(
            float("-inf")
        )
        edge_weight[data.edge_filter.to(torch.bool)] = mask
        argsort = torch.argsort(edge_weight, dim=1, descending=True)
        indices = torch.arange(
            edge_weight.size(1), device=edge_weight.device
        ).expand_as(edge_weight)
        top_edges = indices < top_num_edges.unsqueeze(1)
        new_edge_id = argsort[top_edges]
        new_batch_id = torch.arange(
            edge_weight.size(0), device=edge_weight.device
        ).repeat_interleave(top_num_edges)
        # a new mask only selecting the top edges based on a ratio
        mask = edge_weight[new_batch_id, new_edge_id]
        # get the selected edges
        edges = data.batched_edge_index[:, new_batch_id, new_edge_id]
        edge_type = data.batched_edge_type[new_batch_id, new_edge_id]
        if self.counter_factual_eval:
            data = counter_factual_edge_filter(
                data, batch_id=new_batch_id, edge_id=new_edge_id
            )
        return data, mask, new_batch_id, edges, edge_type

    def select_topk_nodes(self, data, mask, top_num_nodes):
        edge_weight = torch.zeros(data.edge_filter.shape, device=mask.device)
        edge_weight[data.edge_filter.to(torch.bool)] = mask
        in_weight = customized_func.scatter_mean(
            edge_weight,
            data.batched_edge_index[0],
            preserved_edges=data.edge_filter,
            dim=-1,
            dim_size=data.max_num_nodes,
        )
        out_weight = customized_func.scatter_mean(
            edge_weight,
            data.batched_edge_index[1],
            preserved_edges=data.edge_filter,
            dim=-1,
            dim_size=data.max_num_nodes,
        )
        node_weight = (in_weight + out_weight) / 2
        indices = torch.arange(
            node_weight.size(1), device=node_weight.device
        ).expand_as(node_weight)
        valid_nodes = indices < data.subgraph_num_nodes.unsqueeze(1)
        node_weight[~valid_nodes] = float("-inf")
        argsort = torch.argsort(node_weight, dim=1, descending=True)
        top_nodes = indices < top_num_nodes.unsqueeze(1)
        s_node_id = argsort[top_nodes]  # the selected node id within each subgraph
        node_batch = torch.arange(
            node_weight.size(0), device=node_weight.device
        ).repeat_interleave(top_num_nodes)
        # make an offset so that each node_id will be unique
        offsets = node_batch * data.num_nodes
        unique_node_id = s_node_id + offsets
        # get the edges composed of the selected edges
        offsets = (data.edge_batch * data.num_nodes).unsqueeze(0)
        unique_edges = data.edge_index + offsets
        # assert torch.all(torch.isin(unique_node_id, unique_edges.flatten())) Sometimes the selected node is not inside the edges (imagine singleton head)
        edge_mask = torch.all(torch.isin(unique_edges, unique_node_id), dim=0)
        # get the mask for the edges selected
        mask = mask[edge_mask]
        # get the selected edges, and the edge type
        edges = data.edge_index[:, edge_mask]
        edge_type = data.edge_type[edge_mask]
        new_batch_id = data.edge_batch[edge_mask]
        if self.counter_factual_eval:
            data = counter_factual_edge_filter(data, edge_mask=edge_mask)
        return data, mask, new_batch_id, edges, edge_type

    def hard_edge_masks(self, data, mask, top_num_edges):
        """
        For each subgraph, does a hard edge mask to get the topk edges.
        """
        data, mask, new_batch_id, edges, edge_type = self.select_topk_edges(
            data, mask, top_num_edges
        )
        # Only prune the unimportant edges if we are doing factual explanation.
        if self.factual_eval:
            data = recreate_data_object(data, edges, edge_type, new_batch_id)
        node_mask, edge_mask = self.get_node_mask_and_edge_mask(
            data, new_batch_id, edges, edge_type
        )
        if not self.keep_edge_weight:  # the imp edges get a value 1
            mask = mask.fill_(1)
        # compute the num edges and num nodes in the explanatory subgraphs
        num_edges = edge_mask.sum(dim=-1)
        num_nodes = node_mask.sum(dim=-1)

        return (
            data,
            mask,
            node_mask,
            edge_mask,
            num_edges,
            num_nodes,
        )

    def hard_node_masks(self, data, mask, top_num_nodes):
        data, mask, new_batch_id, edges, edge_type = self.select_topk_nodes(
            data, mask, top_num_nodes
        )
        # Only prune the unimportant edges if we are doing factual explanation.
        if self.factual_eval:
            data = recreate_data_object(data, edges, edge_type, new_batch_id)
        node_mask, edge_mask = self.get_node_mask_and_edge_mask(
            data, new_batch_id, edges, edge_type
        )
        if not self.keep_edge_weight:  # the imp edges get a value 1
            mask = mask.fill_(1)
        # compute the num edges and num nodes in the explanatory subgraphs
        num_edges = edge_mask.sum(dim=-1)
        num_nodes = node_mask.sum(dim=-1)

        return (
            data,
            mask,
            node_mask,
            edge_mask,
            num_edges,
            num_nodes,
        )

    @torch.no_grad()
    def transform_mask(self, data, mask):
        """
        transform the soft edge mask to hard/soft masks.
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

            return self.hard_node_masks(data, mask, top_num_nodes)

        elif self.eval_mask_type == "hard_edge_mask_threshold":
            if self.hard_edge_mask_threshold < 0:
                raise ValueError("Please set the corrent threshold for hard edge mask")
            preserved_edges = mask >= self.hard_edge_mask_threshold
            new_batch_id = batch_id[preserved_edges]
            new_edge_id = edge_id[preserved_edges]
            new_mask = mask[preserved_edges]
            if not self.keep_edge_weight:  # the imp edges get a value 1
                new_mask = new_mask.fill_(1)

            if self.use_default_aggr:
                drop_edges = False
            else:
                drop_edges = True

            return (
                new_mask,
                new_batch_id,
                new_edge_id,
            )

        elif self.eval_mask_type == "hard_edge_mask_top_k":
            if self.hard_edge_mask_top_k <= 0:
                raise ValueError(
                    "Please set the corrent number of edges for hard edge mask"
                )
            # clip so that max num edges one can return is the num edges in the subgraph
            top_num_edges = torch.where(
                data.subgraph_num_edges >= self.hard_edge_mask_top_k,
                self.hard_edge_mask_top_k,
                data.subgraph_num_edges,
            )

            return self.hard_edge_masks(data, mask, top_num_edges)

        elif self.eval_mask_type == "hard_edge_mask_top_ratio":
            if self.hard_edge_mask_top_ratio <= 0:
                raise ValueError("Please set the corrent ratio for hard edge mask")

            top_num_edges = (
                data.subgraph_num_edges * self.hard_edge_mask_top_ratio
            ).to(torch.int32)

            return self.hard_edge_masks(data, mask, top_num_edges)

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
