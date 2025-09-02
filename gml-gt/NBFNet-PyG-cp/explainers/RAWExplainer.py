import torch
from torch import nn
from torch.nn import functional as F
import time
from explainers.RWBaseExplainer import RWBaseExplainer


class RAWExplainer(RWBaseExplainer):
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
        rel_emb=False,
        size_reg=1,
        ent_reg=1,
        temp_start=2,
        temp_end=1,
        epochs=100,
        sample_bias=0.5,
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
        random_walk_inf=True,
        spanning_tree=False,
        spanning_tree_node_ratio=0.7,
        with_path_loss=False,
        reg_path_loss=1,
        max_path_length=2,
        num_mlp_layer=1,
        topk_tails=1,
        eval_mask_type="hard_node_mask",
        keep_edge_weight=False,
        joint_training=False,
        use_default_loss=True,
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
            random_walk_loss=random_walk_loss,
            adj_aggr=adj_aggr,
            teleport_prob=teleport_prob,
            rw_topk_node=rw_topk_node,
            reg_loss_inside=reg_loss_inside,
            reg_loss_outside=reg_loss_outside,
            use_teleport_adj=use_teleport_adj,
            max_power_iter=max_power_iter,
            threshold=threshold,
            force_include=force_include,
            edge_random_walk=edge_random_walk,
            rw_topk_edge=rw_topk_edge,
            convergence=convergence,
            spanning_tree=spanning_tree,
            spanning_tree_node_ratio=spanning_tree_node_ratio,
            with_path_loss=with_path_loss,
            reg_path_loss=reg_path_loss,
            max_path_length=max_path_length,
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

        self.rel_emb = rel_emb

        self.temp = [temp_start, temp_end]
        self.sample_bias = sample_bias
        # self.temp_schedule = lambda e: self.temp[0]*((self.temp[1]/self.temp[0])**(e/self.epochs)) # this function will make the temperature LARGER as the epoch progresses
        self.temp_schedule = lambda e: self.temp[0] * (
            (self.temp[0] / self.temp[1]) ** (e / self.epochs)
        )  # this function will give the expected behavior of the temperature

        # loss function
        self.use_default_loss = use_default_loss

        # whether to do random walk at inference to get the explanation
        self.random_walk_inf = random_walk_inf

        if self.rel_emb:
            # Z_i, R_r, Z_j (Z_head, R_query ignored as NBFNet already encodes this on the node embs)
            self.expl_embedding = model_to_explain.node_embedding_size * 3
        else:  # Z_i, Z_j
            self.expl_embedding = model_to_explain.node_embedding_size * 2

        # flag to check whether to project the relation to node emb size
        self.project_relation = False
        if (
            self.rel_emb
            and model_to_explain.node_embedding_size
            != model_to_explain.rel_embedding_size
        ):
            # the rel emb and the node emb sizes are different, project the rel emb to the same size as the node emb size.
            self.rel_projection = nn.Linear(
                model_to_explain.rel_embedding_size,
                model_to_explain.node_embedding_size,
            )
            self.project_relation = True

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

    def compute_mask_loss(self, mask):
        """
        Returns the loss score based on the given mask.
        :param masked_pred: Prediction based on the current explanation
        :param label: The label for the prediction
        :param edge_mask: Current explanaiton
        :return: loss
        """
        if self.use_default_loss:
            return self.regularization_loss(mask)
        else:
            raise NotImplementedError

    def get_embedding(self, data, batch):
        """
        If we are training the expl GNN model, this gets the node embeddings using the expl model.
        """
        # the expl model can update its params
        _, node_embeds, R_embeds = self.expl_model(data, batch, return_emb=True)
        return node_embeds, R_embeds

    def get_mask(self, data, batch, node_embeds, R_embeds, epoch=None):
        """
        This function gets the edge mask.
        """
        # get the embedding from expl gnn if we are using it.
        if self.expl_gnn_model:
            node_embeds, R_embeds = self.get_embedding(data, batch)
        src_node_emb = node_embeds[data.edge_batch, data.edge_index[0]]
        tgt_node_emb = node_embeds[data.edge_batch, data.edge_index[1]]

        if self.project_relation:
            R_embeds = self.rel_projection(R_embeds)

        rel_emb = R_embeds[data.edge_type]
        # we don't care about the query H and R since NBFNet already encodes these information in the node emb
        triple_emb = torch.cat([src_node_emb, tgt_node_emb, rel_emb], 1)
        explanation = self.explainer_model(triple_emb)

        if self.training:
            mask = self._sample_graph(
                explanation,
                temperature=self.temp_schedule(epoch),
                bias=self.sample_bias,
            ).squeeze()
        else:
            mask = self._sample_graph(explanation, training=False).squeeze()
        return mask

    def forward(self, data, batch, node_embeds, R_embeds, epoch=None):

        # get the edge mask
        mask = self.get_mask(data, batch, node_embeds, R_embeds, epoch=epoch)

        assert self.training
        # based on the mask, get the edge weight
        edge_weights = self.get_edge_weight(data, mask)
        masked_preds = {}
        for key, edge_weight in edge_weights.items():
            if not self.use_default_aggr:  # aggregation with regards to the weights
                data.edge_filter = edge_weight  # the edge filter used to calculate the degree info is the edge weight
            masked_pred = self.pred_model(data, batch, edge_weight=edge_weight)
            assert not torch.any(torch.isnan(masked_pred))
            masked_preds[key] = masked_pred

        size_loss, mask_ent_loss = self.compute_mask_loss(mask)
        if self.random_walk_loss:
            if self.edge_random_walk:
                rw_loss = self.do_edge_random_walk(data, mask, return_loss=True)
                path_loss = torch.tensor(0, device=mask.device)
            else:
                rw_loss, path_loss = self.do_random_walk(data, mask, return_loss=True)
        else:
            rw_loss = torch.tensor(0, device=mask.device)
            path_loss = torch.tensor(0, device=mask.device)

        return masked_preds, size_loss, mask_ent_loss, rw_loss, path_loss

    @torch.no_grad()
    def evaluate_mask(self, data, batch, node_embeds, R_embeds):

        if "hard" in self.eval_mask_type and self.random_walk_inf:
            do_random_walk = True
        else:
            do_random_walk = False

        if self.prev_batch is not None and torch.equal(batch, self.prev_batch):
            # if we are running evaluations on the same set of triples as the last time
            mask = self.prev_mask
        else:
            mask = self.get_mask(data, batch, node_embeds, R_embeds)
            self.prev_batch = batch
            self.prev_mask = mask
            if do_random_walk:
                if self.edge_random_walk:
                    mask = self.do_edge_random_walk(data, mask, return_loss=False)
                    self.prev_mask = mask
                else:
                    # reset the node distribution
                    self.reset_node_dist()
                    # compute the node distribution for this batch
                    self.do_random_walk(data, mask, store_walk=True)

        if do_random_walk and not self.edge_random_walk:
            data, mask, node_mask, edge_mask, num_edges, num_nodes = (
                self.do_random_walk(data, mask, store_walk=True)
            )
        else:
            data, mask, node_mask, edge_mask, num_edges, num_nodes = (
                self.transform_mask(data, mask)
            )

        # based on the mask, get the edge weight
        edge_weight = self.get_edge_weight_eval(data, mask)

        if not self.use_default_aggr and (
            self.keep_edge_weight or self.eval_mask_type == "soft_edge_mask"
        ):
            # aggregation with regards to the weights
            if self.factual_eval:
                data.edge_filter = edge_weight  # the edge filter used to calculate the degree info is the edge weight
            if self.counter_factual_eval:
                data.edge_filter[data.counter_factual_filter.to(torch.bool)] = 1 - mask

        masked_pred = self.eval_model(data, batch, edge_weight=edge_weight)
        assert not torch.any(torch.isnan(masked_pred))
        return masked_pred, node_mask, edge_mask, num_edges, num_nodes
