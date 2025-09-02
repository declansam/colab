import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import degree
import copy
import time

class DynamicEgoRelPGExplainer(nn.Module):
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
    def __init__(self, 
                 model_to_explain, 
                 rel_emb=False, 
                 size_reg = 1,
                 ent_reg = 1,
                 temp_start = 2,
                 temp_end = 1,
                 epochs = 100,
                 sample_bias=0.5,
                 num_mlp_layer=1,
                 topk_tails=1,
                 eval_mask_type='hard_node_mask',
                 keep_edge_weight=False,
                 return_detailed_loss=False,
                 joint_training=False,
                 use_default_loss=True,
                 use_default_aggr=False,
                 eval_on_random=False
                 ):
        
        super().__init__()
        self.rel_emb = rel_emb

        self.temp = [temp_start, temp_end]
        self.sample_bias = sample_bias
        self.epochs = epochs
        # self.temp_schedule = lambda e: self.temp[0]*((self.temp[1]/self.temp[0])**(e/self.epochs)) # this function will make the temperature LARGER as the epoch progresses
        self.temp_schedule = lambda e: self.temp[0]*((self.temp[0]/self.temp[1])**(e/self.epochs)) # this function will give the expected behavior of the temperature
        self.reg_coefs = [size_reg, ent_reg] # for the loss

        # how many topk tails of the original prediction should we consider for graph sampling
        self.topk_tails = topk_tails

        # loss function
        self.use_default_loss = use_default_loss

        # TRUE -> Normal Aggregation, False -> aggregation w.r.t. to the weights
        self.use_default_aggr=use_default_aggr

        # evaluation mask settings
        self.eval_mask_type = eval_mask_type
        self.max_num_nodes_in_mask = -1
        self.hard_edge_mask_threshold = -1
        self.hard_edge_mask_top_k = -1
        self.hard_edge_mask_top_ratio = -1
        self.keep_edge_weight=keep_edge_weight # whether to keep the same edge weight
        # if False, the important edges will have edge weight 1
        self.eval_on_random = eval_on_random # whether to evaluate on randomly selected explanations

        self.return_detailed_loss = return_detailed_loss

        self.joint_training = joint_training
        
        self.model_to_explain = model_to_explain
        
        if not self.joint_training:
            for param in self.model_to_explain.parameters():
                param.requires_grad = False

        self.model_to_explain.dynamic_explanation = True

        self.hops = model_to_explain.hops
        if self.rel_emb: # Z_i, R_r, Z_j (Z_head, R_query ignored as NBFNet already encodes this on the node embs)
            self.expl_embedding = model_to_explain.node_embedding_size * 3
        else: # Z_i, Z_j, Z_head
            self.expl_embedding = model_to_explain.node_embedding_size * 3

        self.project_relation = False # flag to check whether to project the relation to node emb size
        if self.rel_emb and model_to_explain.node_embedding_size != model_to_explain.rel_embedding_size:
            # the rel emb and the node emb sizes are different, project the rel emb to the same size as the node emb size.
            self.rel_projection = nn.Linear(model_to_explain.rel_embedding_size, model_to_explain.node_embedding_size)
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
            eps = (bias - (1-bias)) * torch.rand(sampling_weights.size()).to(sampling_weights.device) + (1-bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph = torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        return graph


    def regularization_loss(self, mask):
        """
        Returns the loss score based on the given mask.
        :param masked_pred: Prediction based on the current explanation
        :param label: The label for the prediction
        :param edge_mask: Current explanaiton
        :return: loss
        """
        if self.use_default_loss:
            size_reg = self.reg_coefs[0]
            entropy_reg = self.reg_coefs[1]
            EPS = 1e-15

            # Based on the Loss, we should consider size_loss and mask_ent_loss applied on the node weights.

            # Regularization losses
            # size_loss = torch.sum(mask) * size_reg # SHOULDN'T THIS BE THE MEAN?
            size_loss = torch.mean(mask) * size_reg
            mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS) # the term mask * torch.log(mask) can lead to nan
            mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

            return size_loss, mask_ent_loss
    
        # Calculate size loss that encourages the sum of weights to match the size target
        else:
            raise NotImplementedError
            size_target = mask.size(0)*self.hard_edge_mask_top_ratio # the target size will be determined by the ratio
            size_loss = (torch.sum(mask) - size_target) ** 2

            # Asymmetric sparsity loss to favor pushing weights towards 1 more strongly than towards 0
            alpha_high = 2.0  # Stronger penalty factor for pushing values closer to 1
            alpha_low = 0.5   # Weaker penalty factor for pushing values closer to 0

            '''
            when mask = 0.1
            alpha_high * 0 * 1
            '''

            sparsity_loss = torch.sum(alpha_high * (mask - 1) ** 2 * mask) + torch.sum(alpha_low * mask ** 2 * (1 - mask))

            # Combined size loss
            size_loss = size_reg * size_loss + sparsity_reg * sparsity_loss_component

            mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
            mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

            # Total loss
            return size_loss, mask_ent_loss
        
    @torch.no_grad()
    def full_prediction(self, data, batch):
        self.model_to_explain.eval()
        original_pred, embeds, R = self.model_to_explain(data, batch, return_emb=True)
        return original_pred, embeds, R

        
    def forward(self, data, batch, node_embeds, R_embeds, epoch=None):
        start_time = time.time()
        if not self.joint_training:
            self.model_to_explain.eval()

        src_node_emb = node_embeds[data.edge_batch, data.edge_index[0]]
        tgt_node_emb = node_embeds[data.edge_batch, data.edge_index[1]]

        if self.project_relation:
            R_embeds = self.rel_projection(R_embeds)

        rel_emb = R_embeds[data.edge_type]
        # we don't care about the query H and R since NBFNet already encodes these information in the node emb
        triple_emb = torch.cat([src_node_emb, tgt_node_emb, rel_emb], 1)
        explanation = self.explainer_model(triple_emb)

        if self.training:
            mask = self._sample_graph(explanation, 
                                      temperature = self.temp_schedule(epoch), 
                                      bias=self.sample_bias).squeeze()
            edge_weight = torch.zeros(data.edge_filter.shape, device=mask.device)
            edge_weight[data.edge_filter.to(torch.bool)] = mask
            if not self.use_default_aggr: # aggregation with regards to the weights
                data.edge_filter = edge_weight # the edge filter used to calculate the degree info is the edge weight
            masked_pred = self.model_to_explain(data, batch, edge_weight=edge_weight)
            assert not torch.any(torch.isnan(masked_pred))

        else:
            mask = self._sample_graph(explanation, training=False).squeeze()
            mask, edge_filter, node_mask, num_edges, num_nodes = self.transform_mask(data, mask)
            edge_weight = torch.zeros(edge_filter.shape, device=mask.device)
            edge_weight[edge_filter.to(torch.bool)] = mask
            
            if not self.use_default_aggr and (self.keep_edge_weight or self.eval_mask_type == 'soft_edge_mask'): 
                # aggregation with regards to the weights
                data.edge_filter = edge_weight # the edge filter used to calculate the degree info is the edge weight
            else:
                data.edge_filter = edge_filter
                
            masked_pred = self.model_to_explain(data, batch, edge_weight=edge_weight)
            assert not torch.any(torch.isnan(masked_pred))
            return masked_pred, node_mask, num_edges, num_nodes
        
        size_loss, mask_ent_loss = self.regularization_loss(mask)

        total_time = time.time() - start_time
        # print(f"* Creating K hop took {k_hop_time/total_time*100:.2f}% of forward() *")
        return masked_pred, size_loss, mask_ent_loss
    

    def get_node_mask_from_edges(self, data, batch_id, edges):
        '''
        get the node mask from selected edges
        batch_id: the batch index for each edge
        edges: the selected edges (indices in subgraph node id)
        '''
        device = data.edge_filter.device
        batch_size = data.edge_filter.size(0)
        # construct the node mask (batch_size, num_nodes_full_graph) indicating whether each node is included in explanatory graph
        # first, create a tensor (batch_size, max_num_nodes) that maps from the subgraph node index to full node index
        node_mapping = torch.zeros((batch_size, data.max_num_nodes), device=device, dtype=data.node_id.dtype)
        indices = torch.arange(data.max_num_nodes, device=device).repeat(batch_size, 1)
        node_filter = indices < data.subgraph_num_nodes.unsqueeze(1)
        node_mapping[node_filter] = data.node_id
        # then, from the s_node_id of the edges in the explanatory graph, get the corresponding full graph node_id
        # mark these nodes as True on the node mask
        node_mask = torch.zeros((batch_size, data.num_nodes), device=device, dtype=torch.bool)
        src = edges[0]
        src = node_mapping[batch_id, src]
        node_mask[batch_id, src] = True
        tgt = edges[1]
        tgt = node_mapping[batch_id, tgt]
        node_mask[batch_id, tgt] = True

        return node_mask
    
    def select_topk_edges(self, data, mask, top_num_edges):
        '''get the topk edges for each triple
        '''
        edge_weight = torch.zeros(data.edge_filter.shape, device=mask.device).fill_(float('-inf')) 
        edge_weight[data.edge_filter.to(torch.bool)] = mask
        argsort = torch.argsort(edge_weight, dim=1, descending=True)
        indices = torch.arange(edge_weight.size(1), device=edge_weight.device).expand_as(edge_weight)
        top_edges = indices < top_num_edges.unsqueeze(1)
        new_edge_id = argsort[top_edges]
        new_batch_id = torch.arange(edge_weight.size(0), device=edge_weight.device).repeat_interleave(top_num_edges)
        # new_edge_weight = torch.zeros(k_hop_subgraphs.shape, device=mask.device)
        mask = edge_weight[new_batch_id, new_edge_id] # a new mask only selecting the top edges based on a ratio
        edge_filter = torch.zeros(data.edge_filter.shape, device=mask.device)
        edge_filter[new_batch_id, new_edge_id] = 1
        # get the selected edges
        edges = data.batched_edge_index[:, new_batch_id, new_edge_id]

        return mask, edge_filter, new_batch_id, edges

    def transform_mask(self, data, mask):
        '''
        transform the soft edge mask for evaluation.
        '''
        if self.eval_on_random:
            mask = torch.rand(mask.shape, device=mask.device)

        if self.eval_mask_type == 'hard_node_mask_top_ratio':
            ...

        elif self.eval_mask_type == 'hard_node_mask_top_k':
            if self.hard_node_mask_top_k < 0:
                raise ValueError('Please set the correct top k for hard node mask top k')
            in_degree = degree(data.edge_index[0], data.num_nodes)
            out_degree = degree(data.edge_index[1], data.num_nodes)
            in_weight = torch.zeros(data.num_nodes).to(mask.device)
            out_weight = torch.zeros(data.num_nodes).to(mask.device)
            in_weight.scatter_add(0, data.edge_index[0], mask)
            out_weight.scatter_add(0, data.edge_index[1], mask)
            node_weight = (in_weight/in_degree + out_weight/out_degree)/2
            ...

        elif self.eval_mask_type == 'hard_edge_mask_threshold':
            if self.hard_edge_mask_threshold < 0:
                raise ValueError('Please set the corrent threshold for hard edge mask')
            preserved_edges = mask >= self.hard_edge_mask_threshold
            new_batch_id = batch_id[preserved_edges]
            new_edge_id = edge_id[preserved_edges]
            new_mask = mask[preserved_edges]
            if not self.keep_edge_weight: # the imp edges get a value 1
                new_mask = new_mask.fill_(1)
            
            if self.use_default_aggr:
                drop_edges = False
            else:
                drop_edges = True

            return new_mask, new_batch_id, new_edge_id # the location of the mask changes
        
        elif self.eval_mask_type == 'hard_edge_mask_top_k':
            if self.hard_edge_mask_top_k <= 0:
                raise ValueError('Please set the corrent number of edges for hard edge mask')
            # clip so that max num edges one can return is the num edges in the subgraph
            top_num_edges = torch.where(data.subgraph_num_edges>=self.hard_edge_mask_top_k, self.hard_edge_mask_top_k, data.subgraph_num_edges)
            mask, edge_filter, new_batch_id, edges = self.select_topk_edges(data, mask, top_num_edges) 
            node_mask = self.get_node_mask_from_edges(data, new_batch_id, edges)
            if not self.keep_edge_weight: # the imp edges get a value 1
                mask = mask.fill_(1)
            # compute the num edges and num nodes in the explanatory subgraphs
            num_edges = edge_filter.sum(dim=-1)
            num_nodes = node_mask.sum(dim=-1)

            return mask, edge_filter, node_mask, num_edges, num_nodes # the location of the mask changes

        elif self.eval_mask_type == 'hard_edge_mask_top_ratio':
            if self.hard_edge_mask_top_ratio <= 0:
                raise ValueError('Please set the corrent ratio for hard edge mask')

            top_num_edges = (data.subgraph_num_edges*self.hard_edge_mask_top_ratio).to(torch.int32)
            mask, edge_filter, new_batch_id, edges = self.select_topk_edges(data, mask, top_num_edges) 
            node_mask = self.get_node_mask_from_edges(data, new_batch_id, edges)
            if not self.keep_edge_weight: # the imp edges get a value 1
                mask = mask.fill_(1)
            # compute the num edges and num nodes in the explanatory subgraphs
            num_edges = edge_filter.sum(dim=-1)
            num_nodes = node_mask.sum(dim=-1)

            return mask, edge_filter, node_mask, num_edges, num_nodes # the location of the mask changes

        elif self.eval_mask_type == 'soft_edge_mask':
            node_mask = self.get_node_mask_from_edges(data, data.edge_batch, data.edge_index)
            return mask, data.edge_filter, node_mask, data.subgraph_num_edges, data.subgraph_num_nodes
        
        else:
            raise NotImplementedError