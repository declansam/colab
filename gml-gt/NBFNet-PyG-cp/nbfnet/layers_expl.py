import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter

import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree
from typing import Tuple

from . import customized_func

import logging
import time

logger = logging.getLogger(__file__)


class GeneralizedRelationalConv(MessagePassing):

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    # TODO for compile() - doesn't work currently
    # propagate_type = {"edge_index": torch.LongTensor, "size": Tuple[int, int]}

    def __init__(
        self,
        input_dim,
        output_dim,
        num_relation,
        query_input_dim,
        message_func="distmult",
        aggregate_func="pna",
        layer_norm=False,
        activation="relu",
        dependent=True,
        use_pyg_propagation=False,
    ):
        super(GeneralizedRelationalConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent
        self.use_pyg_propagation = use_pyg_propagation

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)

        if dependent:
            # obtain relation embeddings as a projection of the query relation
            self.relation_linear = nn.Linear(query_input_dim, num_relation * input_dim)
        else:
            # relation embeddings as an independent embedding matrix per each layer
            self.relation = nn.Embedding(num_relation, input_dim)

    def forward(
        self,
        input,
        query,
        boundary,
        edge_index,
        edge_type,
        size,
        batched_edge_index,
        batched_edge_type,
        edge_filter,
        edge_weight=None,
    ):
        batch_size = len(query)

        if self.dependent:
            # layer-specific relation features as a projection of input "query" (relation) embeddings
            relation = self.relation_linear(query).view(
                batch_size, self.num_relation, self.input_dim
            )
        else:
            # layer-specific relation features as a special embedding matrix unique to each layer
            relation = self.relation.weight.expand(batch_size, -1, -1)
        if edge_weight is None:
            edge_weight = edge_filter

        # note that we send the initial boundary condition (node states at layer0) to the message passing
        # correspond to Eq.6 on p5 in https://arxiv.org/pdf/2106.06935.pdf
        output = self.propagate(
            edge_index,
            x=input,
            relation=relation,
            boundary=boundary,
            edge_type=edge_type,
            size=size,
            batched_edge_index=batched_edge_index,
            batched_edge_type=batched_edge_type,
            edge_filter=edge_filter,
            edge_weight=edge_weight,
        )
        return output

    def propagate(self, edge_index, size=None, **kwargs):
        if (
            kwargs["edge_weight"].requires_grad
            or self.message_func == "rotate"
            or kwargs["edge_weight"].ndim == 2
            or self.use_pyg_propagation
        ):
            # the rspmm cuda kernel only works for TransE and DistMult message functions
            # otherwise we invoke separate message & aggregate functions
            return super(GeneralizedRelationalConv, self).propagate(
                edge_index, size, **kwargs
            )

        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        # in newer PyG,
        # __check_input__ -> _check_input()
        # __collect__ -> _collect()
        # __fused_user_args__ -> _fuser_user_args
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._fused_user_args, edge_index, size, kwargs)

        # TODO: use from packaging.version import parse as parse_version as by default 2.4 > 2.14 which is wrong
        # Let's collectively hope there will be PyG 3.0 after 2.9 and not 2.10
        pyg_version = [int(i) for i in torch_geometric.__version__.split(".")]
        col_fn = (
            self.inspector.distribute
            if pyg_version[1] <= 4
            else self.inspector.collect_param_data
        )

        msg_aggr_kwargs = col_fn("message_and_aggregate", coll_dict)
        for hook in self._message_and_aggregate_forward_pre_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs))
            if res is not None:
                edge_index, msg_aggr_kwargs = res
        out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
        for hook in self._message_and_aggregate_forward_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs), out)
            if res is not None:
                out = res

        # PyG 2.5+ distribute -> collect_param_data
        update_kwargs = col_fn("update", coll_dict)
        out = self.update(out, **update_kwargs)

        for hook in self._propagate_forward_hooks.values():
            res = hook(self, (edge_index, size, kwargs), out)
            if res is not None:
                out = res

        return out

    def message(self, batched_edge_index, batched_edge_type, x, relation, boundary):
        # start = time.time()
        # relation_j (B, max_num_edges, 32)
        if batched_edge_index.numel() == 0:
            # if no edges return the same representation
            return boundary

        batch_size, max_num_edges = batched_edge_type.shape
        batch = torch.arange(batch_size).repeat_interleave(max_num_edges)
        relation_j = relation[batch, batched_edge_type.flatten()].view(
            batch_size, max_num_edges, -1
        )

        # message function of the source using edge type
        src_nodes = batched_edge_index[0]
        x_j = x[batch, src_nodes.flatten()].view(batch_size, max_num_edges, -1)
        # num_batch, num_edges, emb
        if self.message_func == "transe":
            message = x_j + relation_j
        elif self.message_func == "distmult":
            message = x_j * relation_j
        elif self.message_func == "rotate":
            x_j_re, x_j_im = x_j.chunk(2, dim=-1)
            r_j_re, r_j_im = relation_j.chunk(2, dim=-1)
            message_re = x_j_re * r_j_re - x_j_im * r_j_im
            message_im = x_j_re * r_j_im + x_j_im * r_j_re
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)

        # augment messages with the boundary condition
        message = torch.cat(
            [message, boundary], dim=self.node_dim
        )  # (batch_size, num_edges + num_nodes, input_dim)

        # logger.warning(f"* MSG Took {(time.time() - start):.3f}s*")
        return message

    def aggregate(self, input, edge_weight, batched_edge_index, dim_size, edge_filter):
        start = time.time()
        batch_size = batched_edge_index.size(1)
        tgt_nodes = batched_edge_index[1]
        # augment aggregation index (tgt node indices) with self-loops for the boundary condition
        index = torch.cat(
            [
                tgt_nodes,
                torch.arange(dim_size, device=input.device).expand(batch_size, -1),
            ],
            dim=-1,
        )  # (batch_size, max_num_edges + max_num_nodes)
        self_loops = torch.ones((batch_size, dim_size), device=input.device)
        edge_filter = torch.cat(
            [edge_filter, self_loops.to(torch.bool)], dim=-1
        ).unsqueeze(-1)
        assert (
            edge_weight.ndim == 2
        )  # we have different edge weight per triple in batch
        edge_weight = torch.cat([edge_weight, self_loops], dim=-1).unsqueeze(-1)
        # the assumption: for edges that are False in edge_filter, edge_weight has to be 0
        assert torch.all(edge_weight[edge_filter == 0] == 0)

        assert edge_filter.dtype != torch.bool and edge_filter.dtype != torch.bool

        # we will be dropping the edges here using edge_filter
        scatter_func = customized_func.customized_scatter
        degree_func = customized_func.customized_degree
        if (
            self.aggregate_func == "pna"
        ):  # if we are dropping edges, we need to have a customized scatter function
            start_mean = time.time()
            mean = scatter_func(
                input * edge_weight,
                index,
                preserved_edges=edge_filter,
                dim=self.node_dim,
                dim_size=dim_size,
                reduce="mean",
            )
            sq_mean = scatter_func(
                input**2 * edge_weight,
                index,
                preserved_edges=edge_filter,
                dim=self.node_dim,
                dim_size=dim_size,
                reduce="mean",
            )
            # logger.warning(f"* AGGR_MEAN Took {(time.time() - start_mean):.3f}s*")
            max = scatter_func(
                input * edge_weight,
                index,
                preserved_edges=edge_filter,
                dim=self.node_dim,
                dim_size=dim_size,
                reduce="max",
            )
            assert not torch.any(
                max == float("-inf")
            )  # we gave -inf as the message of the dropped messages
            min = scatter_func(
                input * edge_weight,
                index,
                preserved_edges=edge_filter,
                dim=self.node_dim,
                dim_size=dim_size,
                reduce="min",
            )
            assert not torch.any(
                min == float("inf")
            )  # we gave inf as the message of the dropped edges
            std = (sq_mean - mean**2).clamp(min=self.eps).sqrt()
            features = torch.cat(
                [
                    mean.unsqueeze(-1),
                    max.unsqueeze(-1),
                    min.unsqueeze(-1),
                    std.unsqueeze(-1),
                ],
                dim=-1,
            )
            features = features.flatten(-2)
            # if degree_by_weights:
            #     degree_out = degree_func(index, preserved_edges=edge_weight, dim=self.node_dim, num_nodes=dim_size).unsqueeze(-1) # customized degree function for dropping edges
            # else:
            degree_out = degree_func(
                index,
                preserved_edges=edge_filter,
                dim=self.node_dim,
                num_nodes=dim_size,
            ).unsqueeze(
                -1
            )  # customized degree function for dropping edges
            assert torch.all(
                degree_out >= 1
            )  # for all the nodes, it should at least have at least 1 degree (self-loop)
            # specifically problematic when ALL the edges are dropped, clamp the mean to eps
            scale = degree_out.log()
            mean = scale.mean(1).unsqueeze(-1)
            mean = torch.clamp(mean, min=self.eps)
            scale = scale / mean
            scales = torch.cat(
                [torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1
            )
            output = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            output = scatter_func(
                input * edge_weight,
                index,
                preserved_edges=edge_filter,
                dim=self.node_dim,
                dim_size=dim_size,
                reduce=self.aggregate_func,
            )

        assert not torch.any(torch.isnan(output))

        # logger.warning(f"* AGGR Took {(time.time() - start):.3f}s*")
        return output

    def message_and_aggregate(
        self,
        edge_index,
        x,
        relation,
        boundary,
        edge_type,
        edge_weight,
        index,
        dim_size,
    ):
        # fused computation of message and aggregate steps with the custom rspmm cuda kernel
        # speed up computation by several times
        # reduce memory complexity from O(|E|d) to O(|V|d), so we can apply it to larger graphs
        from .rspmm import generalized_rspmm

        batch_size, num_node = x.shape[:2]
        x_flat = x.transpose(0, 1).flatten(1)
        relation = relation.transpose(0, 1).flatten(1)
        boundary = boundary.transpose(0, 1).flatten(1)
        degree_out = degree(index, dim_size).unsqueeze(-1) + 1

        if self.message_func in self.message2mul:
            mul = self.message2mul[self.message_func]
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        if self.aggregate_func == "sum":
            update = generalized_rspmm(
                edge_index, edge_type, edge_weight, relation, x_flat, sum="add", mul=mul
            )
            update = update + boundary
        elif self.aggregate_func == "mean":
            update = generalized_rspmm(
                edge_index, edge_type, edge_weight, relation, x_flat, sum="add", mul=mul
            )
            update = (update + boundary) / degree_out
        elif self.aggregate_func == "max":
            update = generalized_rspmm(
                edge_index, edge_type, edge_weight, relation, x_flat, sum="max", mul=mul
            )
            update = torch.max(update, boundary)
        elif self.aggregate_func == "pna":
            # we use PNA with 4 aggregators (mean / max / min / std)
            # and 3 scalars (identity / log degree / reciprocal of log degree)
            sum = generalized_rspmm(
                edge_index, edge_type, edge_weight, relation, input, sum="add", mul=mul
            )
            sq_sum = generalized_rspmm(
                edge_index,
                edge_type,
                edge_weight,
                relation**2,
                input**2,
                sum="add",
                mul=mul,
            )
            max = generalized_rspmm(
                edge_index, edge_type, edge_weight, relation, input, sum="max", mul=mul
            )
            min = generalized_rspmm(
                edge_index, edge_type, edge_weight, relation, input, sum="min", mul=mul
            )
            mean = (sum + boundary) / degree_out
            sq_mean = (sq_sum + boundary**2) / degree_out
            max = torch.max(max, boundary)
            min = torch.min(min, boundary)  # (node, batch_size * input_dim)
            std = (sq_mean - mean**2).clamp(min=self.eps).sqrt()
            features = torch.cat(
                [
                    mean.unsqueeze(-1),
                    max.unsqueeze(-1),
                    min.unsqueeze(-1),
                    std.unsqueeze(-1),
                ],
                dim=-1,
            )
            features = features.flatten(-2)  # (node, batch_size * input_dim * 4)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat(
                [torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1
            )  # (node, 3)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(
                -2
            )  # (node, batch_size * input_dim * 4 * 3)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        update = update.view(num_node, batch_size, -1).transpose(0, 1)
        return update

    def update(self, update, x):
        # node update as a function of old states (x) and this layer output (update)
        output = self.linear(torch.cat([x, update], dim=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output
