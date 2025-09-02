import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import degree
import copy

class RelPGExplainer(nn.Module):
    """
    A class for the Relational-PGExplainer (https://arxiv.org/abs/2011.04573).
    
    :param model_to_explain: graph classification model who's predictions we wish to explain.
    :param rel_emb: whether to use the relation embeddings
    :param temp: the temperture parameters dictacting how we sample our random graphs.
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
                 temp=(5.0, 2.0), 
                 reg_coefs=(0.05, 0.05),
                 epochs = 100,
                 sample_bias=0,
                 num_mlp_layer=1,
                 mask_type='hard_node_mask',
                 max_num_nodes_in_mask=200,
                 return_detailed_loss=False
                 ):
        
        super().__init__()
        self.rel_emb = rel_emb

        self.temp = temp
        self.sample_bias = sample_bias
        self.epochs = epochs
        self.temp_schedule = lambda e: self.temp[0]*((self.temp[1]/self.temp[0])**(e/self.epochs))
        self.reg_coefs = reg_coefs # for the loss

        self.mask_type = mask_type
        self.max_num_nodes_in_mask = max_num_nodes_in_mask
        
        self.return_detailed_loss = return_detailed_loss
        
        self.model_to_explain = model_to_explain
        for param in self.model_to_explain.parameters():
            param.requires_grad = False

        self.hops = model_to_explain.hops
        if self.rel_emb: # Z_i, R_r, Z_j, Z_head, R_query
            self.expl_embedding = model_to_explain.node_embedding_size * 5
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
        :param bias: Bias on the weights to make samplign less deterministic
        :param training: If set to false, the samplign will be entirely deterministic
        :return: sample graph
        """
        if training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1-bias)) * torch.rand(sampling_weights.size()).to(sampling_weights.device) + (1-bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph = torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        return graph


    def loss(self, masked_pred, label, mask):
        """
        Returns the loss score based on the given mask.
        :param masked_pred: Prediction based on the current explanation
        :param label: The label for the prediction
        :param edge_mask: Current explanaiton
        :return: loss
        """
        size_reg = self.reg_coefs[0]
        entropy_reg = self.reg_coefs[1]
        EPS = 1e-15


        # Based on the Loss, we should consider size_loss and mask_ent_loss applied on the node weights.

        # Regularization losses
        # size_loss = torch.sum(mask) * size_reg # SHOULDN'T THIS BE THE MEAN?
        size_loss = torch.mean(mask) * size_reg
        mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS) # the term mask * torch.log(mask) can lead to nan
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

        # Explanation loss
        bce_loss = F.binary_cross_entropy_with_logits(masked_pred, label.to(torch.float32), reduction='mean')
        # kl_div = torch.nn.functional.kl_div(masked_pred, original_pred, reduction='batchmean')
        # cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred)

        if self.return_detailed_loss:
            return bce_loss, size_loss, mask_ent_loss
        
        loss = bce_loss + size_loss + mask_ent_loss
        assert not torch.isnan(loss)
        return loss

    def create_expl_input(self,
                        batch_edge_index, 
                        batch_edge_type, 
                        batch_id, 
                        h_index,
                        r_index,
                        node_embeds, 
                        R):
        '''
        batch_edge_index: (2, total_num_edges) the edges in the ego networks
        batch_edge_type: (total_num_edges) the attr of edges in the ego networks
        batch_id: (total_num_edges) the batch id for each ego network
        h_index: (batch) the head of the batches
        r_index: (batch) the query_rel of the batches
        node_embeds: (total_num_nodes, emb) the embeddings obtained 
        R: (num_rel, emb)
        '''
        rows = batch_edge_index[0]
        cols = batch_edge_index[1]
        row_embeds = node_embeds[rows] # Z_i (total_num_edges, emb)
        col_embeds = node_embeds[cols] # Z_j (total_num_edges, emb)

        if self.project_relation:
            R = self.rel_projection(R)
            
        rel_embeds = R[batch_edge_type] # R_r (total_num_edges, emb)

        head_embeds = node_embeds[h_index] # Z_h (batch, emb)
        head_embeds = head_embeds[batch_id] # Z_h (total_num_edges, emb)
        query_embeds = R[r_index] # R_q (batch, emb)
        query_embeds = query_embeds[batch_id] # R_q (total_num_edges, emb)

        return torch.cat([row_embeds, col_embeds, rel_embeds, head_embeds, query_embeds], 1) # (total_num_edges, emb*5)
    
    def explain(self, data, triples):
        original_pred, embeds, R, heads, rels = self.model_to_explain(data, triples, mode=data.mode, return_emb=True)
        # embeds: (batch_size, max_subgraph_num_nodes, emb_size)
        # R: (num_rels, emb_size)
        node_embeds = embeds[data.batch, data.node_id] # (total_num_nodes, emb_size)
        batch_id = torch.arange(data.num_graphs).to(data.x.device)
        batch_id = torch.repeat_interleave(batch_id, data.num_edges_subgraph)
        input_expl = self.create_expl_input(data.edge_index, 
                                            data.edge_type, 
                                            batch_id,
                                            heads.squeeze(),
                                            rels.squeeze(),
                                            node_embeds, 
                                            R)
        explanation = self.explainer_model(input_expl)
        return original_pred, explanation
    
    def forward(self, data, triples, epoch=None):
        self.model_to_explain.eval()
        original_pred, explanation = self.explain(data, triples)
        if self.training:
            mask = self._sample_graph(explanation, 
                                      temperature = self.temp_schedule(epoch), 
                                      bias=self.sample_bias).squeeze()
        else:
            mask = self._sample_graph(explanation, training=False).squeeze()
            testdata, mask = self.create_hard_mask(data, mask)
            result = self.compute_mrr(testdata, mask)
            return result
        
        masked_pred = self.model_to_explain(data, triples, mode=data.mode, edge_weight=mask)

        label = torch.zeros(data.num_nodes, dtype=torch.bool).to(data.x.device)
        label[data.label_node_index] = 1

        loss = self.loss(masked_pred[data.batch, data.node_id], 
                         label,
                         mask)  
        return loss
    
    @torch.no_grad()
    def compute_mrr(self, data, mask):
        # We have to pad the triples s.t. it becomes (batch_size, padded_num_triples, 3)
        max_subgraph_num_nodes = max(data.num_nodes_subgraph)
        eval_triples = torch.zeros((data.num_graphs, max_subgraph_num_nodes, 3), dtype=torch.int64).to(data.x.device)
        eval_triples[data.batch, data.node_id] = torch.cat([data.eval_edge_index, data.eval_edge_label.unsqueeze(0)]).t()
        
        # get the prediction for each node for each graph
        if self.mask_type=='hard_node_mask':
            pred = self.model_to_explain(data, eval_triples, mode=data.mode)
        elif self.mask_type=='soft_edge_mask':
            pred = self.model_to_explain(data, eval_triples, mode=data.mode, edge_weight=mask)
            
        filtered_pred = pred[data.batch, data.node_id] # get the prediction for each node
        filtered_pred[data.eval_filter_index] = float('-inf') # filter the other positive examples
        
        pred = torch.zeros_like(pred).fill_(float('-inf')) # the padded nodes get -inf score to not interfere with the ranking
        pred[data.batch, data.node_id] = filtered_pred # fill in the filtered prediction
        argsort = torch.argsort(pred, dim=1, descending=True) # sort for each batch
        offsets = torch.cumsum(data.num_nodes_subgraph, dim=0).unsqueeze(1) # create offsets for each batch
        argsort[1:]+=offsets[:-1] # add offset s.t. the indices match

        pos_arg = data.pos_arg_index.unsqueeze(1) # the pos_arg is the true positive if it is included
        pos_arg_rank = (argsort == pos_arg).nonzero()[:, 1]+1 # compute the rank
        ranking = torch.where(data.true_candidate_in_graph, pos_arg_rank, data.full_num_nodes[0])
        
        return ranking, data.true_candidate_in_graph
        
    @torch.no_grad()
    def create_hard_mask(self, data, mask):

        test_data = copy.copy(data)

        # You cannot compute the gradients of hard masks.
        if self.mask_type=='hard_node_mask':
            # for each node, get the in / out_degree
            # importance_node = sum_in_edge_weight / in_degree + sum_out_edge_weight / out_degree
            in_degree = degree(data.edge_index[0], data.num_nodes)
            out_degree = degree(data.edge_index[1], data.num_nodes)
            in_weight = torch.zeros(data.num_nodes).to(mask.device)
            out_weight = torch.zeros(data.num_nodes).to(mask.device)
            in_weight.scatter_add(0, data.edge_index[0], mask)
            out_weight.scatter_add(0, data.edge_index[1], mask)
            node_weight = (in_weight/in_degree + out_weight/out_degree)/2
            node_weight[data.center_node_index] = float('inf') # the center node have to included. All the time!
            # in_weight[edge_index[0][i]] += edge_weight[i]
            # out_weight[edge_index[1][i]] += edge_weight[i]
            
            # we need to sort it for each subgraph in the batch
            padded_node_weight = torch.zeros(data.num_graphs, max(data.num_nodes_subgraph)).fill_(float('-inf')).to(node_weight.device)
            padded_node_weight[data.batch, data.node_id] = node_weight
            padded_argsort = torch.argsort(padded_node_weight, dim=1, descending=True)
            offsets = torch.cumsum(data.num_nodes_subgraph, dim=0).unsqueeze(1)
            padded_argsort[1:]+=offsets[:-1] # add offset s.t. the indices match

            # get the num nodes in each subgraph
            # if the num nodes in each subgraph is less than the max_num_nodes_in_mask, we select all the nodes
            test_data.num_nodes_subgraph = torch.where(data.num_nodes_subgraph < self.max_num_nodes_in_mask, data.num_nodes_subgraph, self.max_num_nodes_in_mask)
            
            # construct the node_filter for the hard mask node selection
            node_id = torch.arange(max(data.num_nodes_subgraph)).expand(test_data.num_graphs, max(data.num_nodes_subgraph)).to(test_data.x.device)
            node_filter = node_id < test_data.num_nodes_subgraph.unsqueeze(1)

            # get the relevant nodes
            selected_nodes = padded_argsort[node_filter]
            selected_nodes = torch.sort(selected_nodes)[0] # sort the selected nodes
            # because of the offsets, it will only sort within each subgraph and not mix nodes across different subgraphs.
            # i.e. the first n1 nodes of selected nodes will come from graph1, the next n2 nodes from graph2, ...
            
            # construct the subgraph, with re-labeling.
            # the graph data
            edge_mask = torch.all(torch.isin(data.edge_index, selected_nodes), dim=0)
            test_data.edge_index = torch.bucketize(test_data.edge_index[:, edge_mask], selected_nodes, right=False) # relabel by bucketizing
            test_data.edge_type = test_data.edge_type[edge_mask]
            test_data.x = data.x[selected_nodes]

            # the evaluation data
            edge_mask = torch.all(torch.isin(data.eval_edge_index, selected_nodes), dim=0)
            test_data.eval_edge_index = torch.bucketize(test_data.eval_edge_index[:, edge_mask], selected_nodes, right=False) # relabel by bucketizing
            test_data.eval_edge_label = test_data.eval_edge_label[edge_mask]

            eval_filter_mask = torch.isin(data.eval_filter_index, selected_nodes)
            test_data.eval_filter_index = data.eval_filter_index[eval_filter_mask]

            # Checking if the true tail was included.
            pos_arg = data.pos_arg_index.unsqueeze(1)
            pos_arg_rank = (padded_argsort == pos_arg).nonzero()[:, 1]+1
            included = pos_arg_rank <= test_data.num_nodes_subgraph
            # see if the true candidate was included in the orig. graph.
            # if it was, check if it's included in the subgraph
            # if not, it will never be included
            test_data.true_candidate_in_graph = torch.where(data.true_candidate_in_graph, included, False)

            # other necessary data structures (node_id, and batch)
            test_data.node_id = node_id[node_filter]
            batch_id = torch.arange(test_data.num_graphs).unsqueeze(1).to(node_id.device)
            node_id.fill_(1)
            test_data.batch = (node_id*batch_id)[node_filter]

            return test_data, mask

        elif self.mask_type=='soft_edge_mask':
            return test_data, mask

        else:
            raise NotImplementedError

