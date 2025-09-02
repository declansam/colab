from math import sqrt
import random
import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils.convert import (
    from_scipy_sparse_matrix,
    to_scipy_sparse_matrix,
)
from tqdm import tqdm

from explainers.BaseExplainer import BaseExplainer
from explainers._util import index_edge, remove_target_edge

"""
This is an adaption of the GNNExplainer of the PyTorch-Lightning library. 

The main similarity is the use of the methods _set_mask and _clear_mask to handle the mask. 
The main difference is the handling of different classification tasks. The original Geometric implementation only works for node 
classification. The implementation presented here also works for graph_classification datasets. 

link: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/gnn_explainer.html
"""


class GNNExplainer(BaseExplainer):
    """
    A class encaptulating the GNNexplainer (https://arxiv.org/abs/1903.03894).

    :param model_to_explain: graph classification model who's predictions we wish to explain.
    :param graphs: the collections of edge_indices representing the graphs
    :param features: the collcection of features for each node in the graphs.
    :param task: str "node" or "graph"
    :param epochs: amount of epochs to train our explainer
    :param lr: learning rate used in the training of the explainer
    :param reg_coefs: reguaization coefficients used in the loss. The first item in the tuple restricts the size of the explainations, the second rescticts the entropy matrix mask.
    :param hops: how many layers of GNN?

    :function __set_masks__: utility; sets learnable mask on the graphs.
    :function __clear_masks__: utility; rmoves the learnable mask.
    :function _loss: calculates the loss of the explainer
    :function explain: trains the explainer to return the subgraph which explains the classification of the model-to-be-explained.
    """

    def __init__(
        self,
        model_to_explain,
        id2entity,
        id2relation,
        device,
        epochs=30,
        lr=0.003,
        reg_coefs=(0.05, 1.0),
        hops=3,
    ):
        super().__init__(model_to_explain, id2entity, id2relation, device)
        self.epochs = epochs
        self.lr = lr
        self.reg_coefs = reg_coefs
        self.hops = hops

    def _set_masks(self, num_nodes, edge_index):
        """
        Inject the explanation maks into the message passing modules.
        :param x: features
        :param edge_index: graph representation
        """
        N, E = num_nodes, edge_index.size(1)

        std = torch.nn.init.calculate_gain("relu") * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)

    def _clear_masks(self):
        """
        Cleans the injected edge mask from the message passing modules. Has to be called before any new sample can be explained.
        """
        self.edge_mask = None

    def _loss(self, masked_pred, original_pred, edge_mask, reg_coefs):
        """
        Returns the loss score based on the given mask.
        :param masked_pred: Prediction based on the current explanation
        :param original_pred: Predicion based on the original graph
        :param edge_mask: Current explanaiton
        :param reg_coefs: regularization coefficients
        :return: loss
        """
        size_reg = reg_coefs[0]
        entropy_reg = reg_coefs[1]
        EPS = 1e-15

        # Regularization losses
        mask = torch.sigmoid(edge_mask)
        # mask = edge_mask
        size_loss = torch.sum(mask) * size_reg
        mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(
            1 - mask + EPS
        )
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

        # Explanation loss
        cce_loss = torch.nn.functional.kl_div(
            masked_pred, original_pred, log_target=True, reduction="batchmean"
        )

        return cce_loss + size_loss + mask_ent_loss

    def prepare(self, data):
        """Nothing is done to prepare the GNNExplainer, this happens at every index"""
        return

    def explain(self, data, batch, triplet):
        """
        Main method to construct the explanation for a given sample. This is done by training a mask such that the masked graph still gives
        the same prediction as the original graph using an optimization approach
        :param index: index of the node/graph that we wish to explain
        :return: explanation graph and edge weights
        """
        if batch.shape[0] != 1:
            raise NotImplementedError("GNNExplainer is not implemented to take a batch")

        # Prepare model for new explanation run
        self.model_to_explain.eval()
        self._clear_masks()

        # decide the head. If tail-batch, it is the src, and if head-batch, it is the tgt
        h_index, t_index, _ = batch.unbind(-1)
        is_tail = (h_index == h_index[0, 0]).all(
            dim=-1
        )  # checks if it's tail-batch by seeing if the src is fixed.
        head = torch.where(is_tail, h_index[0, 0], t_index[0, 0])

        # get the k-hop neighborhood of the graph centered around the head node.
        _, edge_index, _, edge_mask = k_hop_subgraph(head, self.hops, data.edge_index)
        s_data = Data(
            edge_index=edge_index,
            edge_type=data.edge_type[edge_mask],
            split=data.split,
            num_nodes=data.num_nodes,
        )

        # for the training split, perform masking s.t. the correct triple and its inverse is not included in the graph.
        if data.split == "train":
            s_data = remove_target_edge(
                self.num_original_relation,
                s_data,
                triplet[:, 0],
                triplet[:, 1],
                triplet[:, 2],
            )

        # get the prediction of the model given the masked graph.
        with torch.no_grad():
            original_pred = self.model_to_explain(s_data, batch)

        # create a learnable parameter over the edges and optimize the edge mask such that the predictions match
        self._set_masks(data.num_nodes, s_data.edge_index)
        optimizer = Adam([self.edge_mask], lr=self.lr)
        self.edge_mask = self.edge_mask.to(self.device)

        # Start training loop
        for e in range(0, self.epochs):
            optimizer.zero_grad()

            # Sample possible explanation
            masked_pred = self.model_to_explain(
                s_data, batch, edge_weight=torch.sigmoid(self.edge_mask)
            )
            loss = self._loss(
                masked_pred, original_pred, self.edge_mask, self.reg_coefs
            )

            loss.backward()
            optimizer.step()

        # Retrieve final explanation
        mask = torch.sigmoid(self.edge_mask)
        s_data.explanation = mask.detach()
        return s_data
