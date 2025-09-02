import torch
import time
from explainers.RWBaseExplainer import RWBaseExplainer


class RandomWalk(RWBaseExplainer):
    def __init__(
        self,
        model_to_explain,
        adj_aggr="max",
        teleport_prob=0.2,
        use_teleport_adj=True,
        edge_random_walk=False,
        convergence="node",
        invert_edges=False,
        ignore_head=False,
        topk_tails=1,
        eval_mask_type="hard_node_mask",
        ego_network=True,
        factual=True,
        counter_factual=False,
        model_to_evaluate=None,
    ):

        super().__init__(
            model_to_explain,
            adj_aggr=adj_aggr,
            teleport_prob=teleport_prob,
            use_teleport_adj=use_teleport_adj,
            edge_random_walk=edge_random_walk,
            convergence=convergence,
            invert_edges=invert_edges,
            ignore_head=ignore_head,
            topk_tails=topk_tails,
            eval_mask_type=eval_mask_type,
            ego_network=ego_network,
            factual=factual,
            counter_factual=counter_factual,
            model_to_evaluate=model_to_evaluate,
        )

        if self.factual or self.counter_factual:
            print("factual / counter_factual objectives not meaningful for RW")

    def train_mask(self, data, batch):
        mask = torch.ones_like(data.edge_type).to(torch.float32)
        if self.edge_random_walk:
            self.edge_mask = self.do_edge_random_walk(data, mask, return_loss=False)
        else:
            self.do_random_walk(data, mask, store_walk=True)

    @torch.no_grad()
    def evaluate_mask(self, data, batch):
        # evaluate the performance given the computed explanation graphs!
        assert (
            "hard" in self.eval_mask_type
        ), "Only evaluation of hard masks supported for random walks."

        if self.edge_random_walk:
            data, mask, node_mask, edge_mask, num_edges, num_nodes = (
                self.transform_mask(data, self.edge_mask)
            )
        else:
            mask = torch.ones_like(data.edge_type).to(torch.float32)
            data, mask, node_mask, edge_mask, num_edges, num_nodes = (
                self.do_random_walk(data, mask, store_walk=True)
            )

        # based on the mask, get the edge weight
        edge_weight = self.get_edge_weight_eval(data, mask)

        masked_pred = self.eval_model(data, batch, edge_weight=edge_weight)
        assert not torch.any(torch.isnan(masked_pred))
        return masked_pred, node_mask, edge_mask, num_edges, num_nodes
