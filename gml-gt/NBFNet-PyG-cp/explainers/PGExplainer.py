import torch
from torch import nn
from torch.nn import functional as F
import time
from explainers.BaseExplainer import BaseExplainer


class PGExplainer(BaseExplainer):
    """
    A class for the PGExplainer (https://arxiv.org/abs/2011.04573).

    :param model_to_explain: graph classification model who's predictions we wish to explain.
    :param rel_emb: whether to use the relation embeddings
    :param temp: the temperture parameters dictacting how we sample our random graphs.
        - temp_start will be the first temperature at the first epoch
        - temp_end will be the final temperature at the last epoch
        - temperature will keep decreasing over the epochs, making the mask more deterministic.
    :param reg_coefs: reguaization coefficients used in the loss. The first item in the tuple restricts the size of the explainations, the second rescticts the entropy matrix mask.
    :params sample_bias: the bias we add when sampling random graphs.
    """

    def __init__(
        self,
        model_to_explain,
        size_reg=1,
        ent_reg=1,
        temp_start=2,
        temp_end=1,
        epochs=100,
        sample_bias=0.5,
        num_mlp_layer=1,
        topk_tails=1,
        eval_mask_type="hard_node_mask",
        keep_edge_weight=False,
        joint_training=False,
        use_default_aggr=False,
        mode="transductive",
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

        # assume we are getting the explanation for the top predicted tail
        assert self.topk_tails == 1

        self.mode = mode
        self.temp = [temp_start, temp_end]
        self.sample_bias = sample_bias
        # self.temp_schedule = lambda e: self.temp[0]*((self.temp[1]/self.temp[0])**(e/self.epochs)) # this function will make the temperature LARGER as the epoch progresses
        self.temp_schedule = lambda e: self.temp[0] * (
            (self.temp[0] / self.temp[1]) ** (e / self.epochs)
        )  # this function will give the expected behavior of the temperature
        # Z_i, Z_j, Z_head, Z_tail
        self.expl_embedding = model_to_explain.node_embedding_size * 4
        # Creation of the explainer_model is done here to make sure that the seed is set
        mlp = []
        for _ in range(num_mlp_layer - 1):
            mlp.append(nn.Linear(self.expl_embedding, self.expl_embedding))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(self.expl_embedding, 1))
        self.explainer_model = nn.Sequential(*mlp)

    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        """
        Implementation of the reparamerization trick to obtain a sample graph while maintaining the posibility to backprop.
        :param sampling_weights: Weights provided by the mlp
        :param temperature: annealing temperature to make the procedure more deterministic
        :param bias: Bias on the weights to make sampling less deterministic
        bias from (0, 1)
        - values close to 0 or 1 will make eps uniformly distributed in [0, 1).
        - values close to 0.5 will make eps constant with 0.5
        - how far away the value is from 0.5 determines the range of the uniform
        - bias = 0.5 -> eps constant 0.5
        - bias = 0.7 -> eps Unifrom [0.3, 0.7) etc...
        - bias = 0 or 1 -> eps Uniform[0, 1)

        When range of eps is smaller and closer to 0.5, it will simulate something closer to a sigmoid
        When range of eps is large, the threshold point can change a lot and make it difficult for the model to learn.

        When temp is 1, it will be the same as a sigmoid, as temp -> 0, it will make it steeper.
        :param training: If set to false, the samplign will be entirely deterministic
        :return: sample graph
        """
        if training:
            if bias == 0:
                bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(sampling_weights.size()).to(
                sampling_weights.device
            ) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph = torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        return graph

    def get_mask(self, data, node_embeds, epoch=None):
        """
        This function gets the edge mask.
        """
        src_node_emb = node_embeds[data.edge_batch, data.edge_index[0]]
        tgt_node_emb = node_embeds[data.edge_batch, data.edge_index[1]]
        head_index, tail_index = data.central_node_index.T
        head_index = head_index.repeat_interleave(data.subgraph_num_edges)
        tail_index = tail_index.repeat_interleave(data.subgraph_num_edges)
        head_node_emb = node_embeds[data.edge_batch, head_index]
        tail_node_emb = node_embeds[data.edge_batch, tail_index]

        edge_emb = torch.cat(
            [src_node_emb, tgt_node_emb, head_node_emb, tail_node_emb], 1
        )
        explanation = self.explainer_model(edge_emb)

        if self.training:
            mask = self._sample_graph(
                explanation,
                temperature=self.temp_schedule(epoch),
                bias=self.sample_bias,
            ).squeeze()

        else:
            mask = self._sample_graph(explanation, training=False).squeeze()
        return mask

    def forward(self, data, batch, node_embeds, epoch=None):

        # get the edge mask
        mask = self.get_mask(data, node_embeds, epoch=epoch)

        if self.training:
            # based on the mask, get the edge weight
            edge_weights = self.get_edge_weight(data, mask)

            if self.factual:
                edge_weight = edge_weights["factual"]
            else:
                edge_weight = edge_weights["counter_factual"]

            if not self.use_default_aggr:  # aggregation with regards to the weights
                data.edge_filter = edge_weight  # the edge filter used to calculate the degree info is the edge weight

            masked_pred = self.pred_model(data, batch, edge_weight=edge_weight)
            assert not torch.any(torch.isnan(masked_pred))

        else:
            data, mask, node_mask, edge_mask, num_edges, num_nodes = (
                self.transform_mask(data, mask)
            )
            # based on the mask, get the edge weight
            edge_weight = self.get_edge_weight_eval(data, mask)

            masked_pred = self.eval_model(data, batch, edge_weight=edge_weight)
            assert not torch.any(torch.isnan(masked_pred))
            return masked_pred, node_mask, edge_mask, num_edges, num_nodes

        size_loss, mask_ent_loss = self.regularization_loss(mask)

        return masked_pred, size_loss, mask_ent_loss

    @torch.no_grad()
    def evaluate_mask(self, data, batch, node_embeds):
        if self.prev_batch is not None and torch.equal(batch, self.prev_batch):
            # if we are running evaluations on the same set of triples as the last time
            mask = self.prev_mask
        else:
            mask = self.get_mask(data, node_embeds)
            self.prev_batch = batch
            self.prev_mask = mask

        data, mask, node_mask, edge_mask, num_edges, num_nodes = self.transform_mask(
            data, mask
        )

        # based on the mask, get the edge weight
        edge_weight = self.get_edge_weight_eval(data, mask)

        masked_pred = self.eval_model(data, batch, edge_weight=edge_weight)
        assert not torch.any(torch.isnan(masked_pred))
        return masked_pred, node_mask, edge_mask, num_edges, num_nodes
