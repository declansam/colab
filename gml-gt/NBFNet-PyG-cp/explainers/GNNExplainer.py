import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import degree
import copy
import time
from torch import optim
from explainers.BaseExplainer import BaseExplainer
from tqdm import tqdm


class GNNExplainer(BaseExplainer):
    """
    A class for the Relational-PGExplainer (https://arxiv.org/abs/2011.04573).

    :param model_to_explain: graph classification model who's predictions we wish to explain.
    :param size_reg: Restricts the size of the explainations.
    :param ent_reg: Rescticts the entropy matrix mask.
    :param epochs: Number of epochs to train the GNNExplainer
    :param eval_mask_type: The mask type used for evaluation
    :param return_detailed_loss: whether to return detailed loss
    :param joint_training: whether to conduct joint training of the GNN.

    """

    def __init__(
        self,
        model_to_explain,
        lr=0.003,
        optimizer="Adam",
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
            model_to_evaluate=model_to_evaluate,
        )

        self.lr = lr
        self.optimizer = optimizer
        assert self.factual + self.counter_factual == 1

    def _set_masks(self, data):
        """
        Inject the explanation maks into the message passing modules.
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

    def train_mask(self, data, batch):
        self._clear_masks()
        self._set_masks(data)

        optimizer = getattr(optim, self.optimizer)([self.edge_mask], lr=self.lr)

        # Start training loop
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
            size_loss, mask_ent_loss = self.regularization_loss(mask)

            if self.factual:  # factual explanation
                label = torch.ones_like(masked_pred)
            else:
                label = torch.zeros_like(masked_pred)  # counter factual

            loss = F.binary_cross_entropy_with_logits(
                masked_pred, label, reduction="mean"
            )
            loss = loss + size_loss + mask_ent_loss

            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def evaluate_mask(self, data, batch):
        # Retrieve final explanation
        mask = torch.sigmoid(self.edge_mask)
        data, mask, node_mask, edge_mask, num_edges, num_nodes = self.transform_mask(
            data, mask
        )
        # based on the mask, get the edge weight
        edge_weight = self.get_edge_weight_eval(data, mask)

        masked_pred = self.eval_model(data, batch, edge_weight=edge_weight)
        assert not torch.any(torch.isnan(masked_pred))

        return masked_pred, node_mask, edge_mask, num_edges, num_nodes
