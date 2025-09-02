import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from explainers.PaGELink import PaGELink, remove_edges_except_k_core_graph
from explainers.data_util import convert_multigraph_to_digraph
from explainers.path_utils import *


def get_inv_path_mask(mask):
    """
    Compute the inverse path mask
    """
    EPS = 1e-15
    inv_mask = 1 / (mask + EPS)
    return inv_mask


def get_path_loss(
    data, edge_index, edge_batch, path_mask, max_path_length, dense=False
):
    """Compute the path loss. This does it per subgraph since backprop using sparse matrix multiplication explodes the memory anyways."""
    EPS = 1e-15

    # get the adj matrix of the edge weights
    batch_size = data.subgraph_num_nodes.size(0)

    total_loss = 0
    for i in range(batch_size):
        # get the edges for the i-th subgraph
        num_nodes = data.subgraph_num_nodes[i]
        edge_mask = edge_batch == i
        s_edge_index = edge_index[:, edge_mask]
        s_path_mask = path_mask[edge_mask]

        if dense:
            mask_adj = torch.zeros((num_nodes, num_nodes), device=s_path_mask.device)
            mask_adj[s_edge_index[0], s_edge_index[1]] = s_path_mask
            adj = torch.zeros_like(mask_adj)
            adj[s_edge_index[0], s_edge_index[1]] = torch.ones_like(s_path_mask)

        else:
            # create the mask adj matrix
            mask_adj = torch.sparse_coo_tensor(
                s_edge_index, s_path_mask, size=(num_nodes, num_nodes)
            )
            mask_adj = mask_adj.coalesce()
            # the adjacency matrix
            adj = torch.sparse_coo_tensor(
                s_edge_index,
                torch.ones_like(s_path_mask),
                size=(num_nodes, num_nodes),
            )
            adj = adj.coalesce()

        # create dense matrix for head nodes
        s_head_index = data.central_node_index[:, 0][i]
        s_tail_index = data.central_node_index[:, 1:][i]
        s_tail_index = s_tail_index.flatten()
        # get the out_edges from the head nodes
        head_edge_mask = torch.isin(s_edge_index[0], s_head_index)
        head_edge_index = s_edge_index[:, head_edge_mask]
        head_path_mask = s_path_mask[head_edge_mask]
        # create the dense matrix for head nodes
        head_mask_adj = torch.zeros((1, mask_adj.size(1)), device=mask_adj.device)
        head_mask_adj[0, head_edge_index[1]] = head_path_mask
        head_adj = torch.zeros((1, mask_adj.size(1)), device=mask_adj.device)
        head_adj[0, head_edge_index[1]] = torch.ones_like(head_path_mask)

        # the power iteration
        loss_on_path = 0
        for i in range(1, max_path_length + 1):
            # the 1 hop paths
            if i == 1:
                aggr_weight = head_mask_adj
                norm = head_adj.clamp(min=1)
                weight = aggr_weight / norm
            else:
                if i != 2:
                    if dense:
                        mask_adj = torch.mm(mask_adj, mask_adj)
                        adj = torch.mm(adj, adj)
                    else:
                        mask_adj = torch.sparse.mm(mask_adj, mask_adj)
                        adj = torch.sparse.mm(adj, adj)
                if dense:
                    aggr_weight = torch.mm(head_mask_adj, mask_adj)
                else:
                    aggr_weight = torch.sparse.mm(head_mask_adj, mask_adj)
                norm = torch.sparse.mm(head_adj, adj).clamp(min=1)
                weight = torch.pow(aggr_weight / norm + EPS, 1 / i)

            path_weight = weight[0, s_tail_index]
            path_weight = path_weight.sum()
            loss_on_path += path_weight
        total_loss += loss_on_path

    if max_path_length > 1:
        total_loss = 1 / (max_path_length - 1) * total_loss
    total_loss = -torch.log(total_loss + EPS)

    return total_loss


class PowerLink(PaGELink):
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
        timeout_duration=3,
        num_mlp_layer=1,
        adj_aggr="max",
        max_path_length=2,
        dense=False,
        size_reg=1,
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
            lr=lr,
            optimizer=optimizer,
            epochs=epochs,
            prune_graph=prune_graph,
            prune_max_degree=prune_max_degree,
            k_core=k_core,
            with_path_loss=with_path_loss,
            timeout_duration=timeout_duration,
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
        self.size_reg = size_reg
        # * Triplet Edge Scorer *
        self.expl_embedding = model_to_explain.node_embedding_size * 6
        # flag to check whether to project the relation to node emb size
        self.project_relation = False
        if model_to_explain.node_embedding_size != model_to_explain.rel_embedding_size:
            # the rel emb and the node emb sizes are different, project the rel emb to the same size as the node emb size.
            self.rel_projection = nn.Linear(
                model_to_explain.rel_embedding_size,
                model_to_explain.node_embedding_size,
            )
            self.project_relation = True

        self.num_mlp_layer = num_mlp_layer
        self.adj_aggr = adj_aggr
        assert (
            self.adj_aggr == "max"
        ), "Only max adj aggr supported for identification of the corresponding edge type"
        self.max_path_length = max_path_length
        self.dense = dense

    def _set_masks(self):
        """
        Create the triplet edge scorer
        """
        mlp = []
        for _ in range(self.num_mlp_layer - 1):
            mlp.append(nn.Linear(self.expl_embedding, self.expl_embedding))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(self.expl_embedding, 1))
        self.explainer_model = nn.Sequential(*mlp).to(
            next(iter(self.pred_model.parameters())).device
        )

    def _clear_masks(self):
        """
        Cleans the injected edge mask from the message passing modules. Has to be called before any new sample can be explained.
        """
        self.explainer_model = None

    def path_loss(self, data, mask):
        # convert the multigraph to digraph
        edge_index, edge_batch, path_mask = convert_multigraph_to_digraph(
            data, mask, self.adj_aggr, handle_singletons=False
        )
        return get_path_loss(
            data, edge_index, edge_batch, path_mask, self.max_path_length, self.dense
        )

    @torch.no_grad()
    def get_paths(self, data, mask):
        """A postprocessing step that turns the mask into actual paths.
        This will continue exploring the paths until the max_budget (either num nodes or num edges)
        is met. In some cases, the loop will take a long time since there are many paths to explore.
        To counter this, we can set a timeout duration for each (src, tgt) pair.
        """
        # Retrieve final explanation
        self.eval()
        # inverse path mask so the highest weight becomes the lowest value
        inv_path_mask = get_inv_path_mask(mask)
        # make the multigraph into a digraph
        path_data, path_mask, inv_path_mask, _ = multigraph_to_graph(
            data, mask, inv_path_mask
        )
        # find the paths
        path_collection = sequential_path_finder_until_budget(
            path_data,
            weight=inv_path_mask,
            return_paths=True,
            eval_mask_type=self.eval_mask_type,
            max_budget=getattr(self, self.eval_mask_type),
            timeout_duration=self.timeout_duration,
        )

        self.path_collection = path_collection
        self.path_data = path_data
        self.path_mask = path_mask

    def triplet_edge_scorer(self, data, batch, node_embeds, R_embeds):

        if self.project_relation:
            R_embeds = self.rel_projection(R_embeds)

        # the edge embeddings
        src_node_emb = node_embeds[data.edge_batch, data.edge_index[0]]
        tgt_node_emb = node_embeds[data.edge_batch, data.edge_index[1]]
        rel_emb = R_embeds[data.edge_type]

        # the target edge embedding
        head_index, tail_index = data.central_node_index.T
        head_index = head_index.repeat_interleave(data.subgraph_num_edges)
        tail_index = tail_index.repeat_interleave(data.subgraph_num_edges)
        _, _, r_index = batch.unbind(-1)
        assert (r_index[:, [0]] == r_index).all()
        r_index = r_index[:, 0].repeat_interleave(data.subgraph_num_edges)
        head_node_emb = node_embeds[data.edge_batch, head_index]
        tail_node_emb = node_embeds[data.edge_batch, tail_index]
        r_emb = R_embeds[r_index]

        triple_emb = torch.cat(
            [
                src_node_emb,
                tgt_node_emb,
                rel_emb,
                head_node_emb,
                tail_node_emb,
                r_emb,
            ],
            1,
        )
        explanation = self.explainer_model(triple_emb)

        return explanation.squeeze()

    def train_mask(self, data, batch, node_embeds, R_embeds):

        self._clear_masks()
        # If counter factual explanation, we don't prune it.
        if self.prune_graph and not self.counter_factual:
            data = self._prune_graph(data)
        self._set_masks()

        optimizer = getattr(optim, self.optimizer)(
            self.explainer_model.parameters(), lr=self.lr
        )

        # Start training loop
        eweight_norm = 0
        EPS = 1e-3
        for e in range(0, self.epochs):
            optimizer.zero_grad()
            assert not torch.any(
                torch.isnan(next(iter(self.explainer_model.parameters())))
            )
            mask = self.triplet_edge_scorer(data, batch, node_embeds, R_embeds)
            mask = torch.sigmoid(mask)

            edge_weights = self.get_edge_weight(data, mask)

            if self.factual:
                edge_weight = edge_weights["factual"]
            else:
                edge_weight = edge_weights["counter_factual"]

            if not self.use_default_aggr:  # aggregation with regards to the weights
                data.edge_filter = edge_weight  # the edge filter used to calculate the degree info is the edge weight

            masked_pred = self.pred_model(data, batch, edge_weight=edge_weight)
            assert not torch.any(torch.isnan(masked_pred))
            size_loss, _ = self.regularization_loss(mask)

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

            loss = loss + path_loss + size_loss
            # loss = loss + size_loss
            self.all_loss["total_loss"] += [loss.item()]

            loss.backward()
            optimizer.step()

        mask = self.triplet_edge_scorer(data, batch, node_embeds, R_embeds)
        mask = torch.sigmoid(mask)

        # after training, compute the explanation paths!
        if "hard" in self.eval_mask_type:
            self.get_paths(data, mask)

    """
    Deprecated Code that computes the path loss in a full-batch manner. Even if we use sparse matrix multiplication, when calling loss.backward()
    the memory usage explodes, likely due to the unavailability of sparse back prop mechanism. Tried torch_sparse as an alternative, but their
    spspmm function does not support backprop. 
    """

    def path_loss_orig(self, data, mask):
        """Compute the path loss."""
        EPS = 1e-15
        # convert the multigraph to digraph
        edge_index, edge_batch, path_mask = convert_multigraph_to_digraph(
            data, mask, self.adj_aggr, handle_singletons=False
        )

        # get the adj matrix of the edge weights
        batch_size = data.subgraph_num_nodes.size(0)
        batch_offsets = torch.cumsum(data.subgraph_num_nodes, dim=0)
        batch_offsets -= data.subgraph_num_nodes
        offsets = batch_offsets[edge_batch]
        # give offsets so each node gets a unique id
        edge_index = edge_index + offsets
        # the mask adjacency matrix
        total_num_nodes = data.subgraph_num_nodes.sum()
        """
        mask_adj = torch.zeros(
            (total_num_nodes, total_num_nodes), device=path_mask.device
        )
        adj = torch.zeros_like(mask_adj)
        mask_adj[edge_index[0], edge_index[1]] = path_mask
        adj[edge_index[0], edge_index[1]] = torch.ones_like(path_mask)
        """
        mask_adj = torch.sparse_coo_tensor(
            edge_index, path_mask, size=(total_num_nodes, total_num_nodes)
        )
        mask_adj = mask_adj.coalesce()
        # the adjacency matrix
        adj = torch.sparse_coo_tensor(
            edge_index,
            torch.ones_like(path_mask),
            size=(total_num_nodes, total_num_nodes),
        )
        adj = adj.coalesce()

        # create dense matrix for head nodes
        head_index = data.central_node_index[:, 0].clone()
        tail_index = data.central_node_index[:, 1:].clone()
        # add offsets
        head_index += batch_offsets
        tail_index += batch_offsets.unsqueeze(1).expand_as(tail_index)
        tail_batch_index = torch.arange(batch_size).unsqueeze(1).expand_as(tail_index)
        tail_index = tail_index.flatten()
        tail_batch_index = tail_batch_index.flatten()
        # get the out_edges from the head nodes
        head_edge_mask = torch.isin(edge_index[0], head_index)
        head_edge_index = edge_index[:, head_edge_mask]
        head_path_mask = path_mask[head_edge_mask]
        # bucketize the head indices so that it takes values (0, ..., batch_size-1)
        head_edge_index[0] = torch.bucketize(
            head_edge_index[0], head_index, right=False
        )
        # create the dense matrix for head nodes
        head_mask_adj = torch.zeros(
            (batch_size, mask_adj.size(1)), device=mask_adj.device
        )
        head_mask_adj[head_edge_index[0], head_edge_index[1]] = head_path_mask
        head_adj = torch.zeros((batch_size, mask_adj.size(1)), device=mask_adj.device)
        head_adj[head_edge_index[0], head_edge_index[1]] = torch.ones_like(
            head_path_mask
        )

        # the power iteration
        loss_on_path = 0
        for i in range(1, self.max_path_length + 1):
            if i == 1:
                aggr_weight = head_mask_adj
                norm = head_adj.clamp(min=1)
                weight = aggr_weight / norm
            else:
                if i != 2:
                    mask_adj = torch.sparse.mm(mask_adj, mask_adj)
                    adj = torch.sparse.mm(adj, adj)
                    """
                    mask_adj = torch.mm(mask_adj, mask_adj)
                    adj = torch.mm(adj, adj)
                    """

                aggr_weight = torch.sparse.mm(head_mask_adj, mask_adj)
                norm = torch.sparse.mm(head_adj, adj).clamp(min=1)
                """
                aggr_weight = torch.mm(head_mask_adj, mask_adj)
                norm = torch.mm(head_adj, adj).clamp(min=1)
                """

                weight = torch.pow(aggr_weight / norm + EPS, 1 / i)

            path_weight = weight[tail_batch_index, tail_index]
            path_weight = path_weight.sum()
            loss_on_path += path_weight

        if self.max_path_length > 1:
            loss_on_path = 1 / (self.max_path_length - 1) * loss_on_path
        loss_on_path = -torch.log(loss_on_path + EPS)

        return loss_on_path
