

'''
forward() from torch_geometric.nn.GATConv
'''

def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, 
            edge_attr: OptTensor = None, size: Size = None,
            return_attention_weights=None):
    # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, NoneType) -> Tensor  # noqa
    # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, NoneType) -> Tensor  # noqa
    # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
    # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
    r"""Runs the forward pass of the module.

    Args:
        return_attention_weights (bool, optional): If set to :obj:`True`,
            will additionally return the tuple
            :obj:`(edge_index, attention_weights)`, holding the computed
            attention weights for each edge. (default: :obj:`None`)
    """
    # NOTE: attention weights will be returned whenever
    # `return_attention_weights` is set to a value, regardless of its
    # actual value (might be `True` or `False`). This is a current somewhat
    # hacky workaround to allow for TorchScript support via the
    # `torch.jit._overload` decorator, as we can only change the output
    # arguments conditioned on type (`None` or `bool`), not based on its
    # actual value.

    H, C = self.heads, self.out_channels

    # We first transform the input node features. If a tuple is passed, we
    # transform source and target node features via separate weights:
    if isinstance(x, Tensor):
        assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
        x_src = x_dst = self.lin_src(x).view(-1, H, C) # (num_nodes, num_heads, out_channel)
    else:  # Tuple of source and target node features:
        x_src, x_dst = x
        assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
        x_src = self.lin_src(x_src).view(-1, H, C)
        if x_dst is not None:
            x_dst = self.lin_dst(x_dst).view(-1, H, C)

    x = (x_src, x_dst)

    # Next, we compute node-level attention coefficients, both for source
    # and target nodes (if present):
    alpha_src = (x_src * self.att_src).sum(dim=-1) # (num_nodes, num_heads)
    alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1) # (num_nodes, num_heads)
    alpha = (alpha_src, alpha_dst)

    if self.add_self_loops: # add self loop to the attention
        if isinstance(edge_index, Tensor):
            # We only want to add self-loops for nodes that appear both as
            # source and target nodes:
            num_nodes = x_src.size(0)
            if x_dst is not None:
                num_nodes = min(num_nodes, x_dst.size(0))
            num_nodes = min(size) if size is not None else num_nodes
            edge_index, edge_attr = remove_self_loops(
                edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, fill_value=self.fill_value,
                num_nodes=num_nodes)
        elif isinstance(edge_index, SparseTensor):
            if self.edge_dim is None:
                edge_index = torch_sparse.set_diag(edge_index)
            else:
                raise NotImplementedError(
                    "The usage of 'edge_attr' and 'add_self_loops' "
                    "simultaneously is currently not yet supported for "
                    "'edge_index' in a 'SparseTensor' form")

    # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
    alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr) # See Point 1

    # alpha (num_edges, num_heads)

    # propagate_type: (x: OptPairTensor, alpha: Tensor)
    out = self.propagate(edge_index, x=x, alpha=alpha, size=size) # See Point 4

    if self.concat:
        out = out.view(-1, self.heads * self.out_channels)
    else:
        out = out.mean(dim=1)

    if self.bias is not None:
        out = out + self.bias

    if isinstance(return_attention_weights, bool):
        if isinstance(edge_index, Tensor):
            if is_torch_sparse_tensor(edge_index):
                # TODO TorchScript requires to return a tuple
                adj = set_sparse_value(edge_index, alpha)
                return out, (adj, alpha)
            else:
                return out, (edge_index, alpha)
        elif isinstance(edge_index, SparseTensor):
            return out, edge_index.set_value(alpha, layout='coo')
    else:
        return out
    
# Point 1: edge_updater from torch_geometric.nn.conv.MessagePassing

def edge_updater(self, edge_index: Adj, **kwargs):
    r"""The initial call to compute or update features for each edge in the
    graph.

    Args:
        edge_index (torch.Tensor or SparseTensor): A :obj:`torch.Tensor`, a
            :class:`torch_sparse.SparseTensor` or a
            :class:`torch.sparse.Tensor` that defines the underlying graph
            connectivity/message passing flow.
            See :meth:`propagate` for more information.
        **kwargs: Any additional data which is needed to compute or update
            features for each edge in the graph.
    """
    ...

    # self._edge_user_args -> {'alpha_i', 'alpha_j', 'edge_attr'}
    coll_dict = self._collect(self._edge_user_args, edge_index, size, # see point 2
                                kwargs)

    edge_kwargs = self.inspector.distribute('edge_update', coll_dict) # collect the arguments for edge_update()
    out = self.edge_update(**edge_kwargs) # see point 3

    ...

    return out

# point 2: _collect from torch_geometric.nn.conv.MessagePassing

def _collect(self, args, edge_index, size, kwargs):
    i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1) # i -> target, j -> source

    out = {}
    for arg in args: # {'alpha_i', 'alpha_j', 'edge_attr'}
        if arg[-2:] not in ['_i', '_j']:
            out[arg] = kwargs.get(arg, Parameter.empty)
        else:
            dim = j if arg[-2:] == '_j' else i
            data = kwargs.get(arg[:-2], Parameter.empty)

            if isinstance(data, (tuple, list)):
                assert len(data) == 2
                if isinstance(data[1 - dim], Tensor):
                    self._set_size(size, 1 - dim, data[1 - dim])
                data = data[dim]

            if isinstance(data, Tensor):
                self._set_size(size, dim, data)
                data = self._lift(data, edge_index, dim)

            out[arg] = data
        '''
        when arg: 'alpha_i' (the alpha of target)
        - dim := 1
        - data := the values of alpha_dst for indicies in edge_index[1]
        
        when arg: 'alpha_j' (the alpha of source)
        - dim := the values of alpha_src for indicies in edge_index[0]

        edge_attr: None
        '''

    ...

    elif isinstance(edge_index, Tensor):
        out['adj_t'] = None
        out['edge_index'] = edge_index
        out['edge_index_i'] = edge_index[i]
        out['edge_index_j'] = edge_index[j]
        out['ptr'] = None

    ...

    out['index'] = out['edge_index_i']
    out['size'] = size
    out['size_i'] = size[i] if size[i] is not None else size[j]
    out['size_j'] = size[j] if size[j] is not None else size[i]
    out['dim_size'] = out['size_i']

    return out

# Point 3
def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
    # Given edge-level attention coefficients for source and target nodes,
    # we simply need to sum them up to "emulate" concatenation:
    alpha = alpha_j if alpha_i is None else alpha_j + alpha_i # sum the attention coefs for source and target nodes (num_edges, num_heads) represents the interaction between u and v for each edge
    if index.numel() == 0:
        return alpha
    if edge_attr is not None and self.lin_edge is not None:
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)
        edge_attr = self.lin_edge(edge_attr)
        edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
        alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
        alpha = alpha + alpha_edge

    alpha = F.leaky_relu(alpha, self.negative_slope)
    alpha = softmax(alpha, index, ptr, size_i) # compute the softmax for each target node
    alpha = F.dropout(alpha, p=self.dropout, training=self.training)
    return alpha

def propagate(self, edge_index: Adj, size: Size = None, **kwargs):

    ...

    for i in range(decomposed_layers):
        ...

        coll_dict = self._collect(self._user_args, edge_index, size, # self._user_args
                                    kwargs)
        
        '''
        get x_j, which is the values of x_src for indices in edge_index[0]
        '''

        msg_kwargs = self.inspector.distribute('message', coll_dict)
        for hook in self._message_forward_pre_hooks.values():
            res = hook(self, (msg_kwargs, ))
            if res is not None:
                msg_kwargs = res[0] if isinstance(res, tuple) else res
        out = self.message(**msg_kwargs)
        for hook in self._message_forward_hooks.values():
            res = hook(self, (msg_kwargs, ), out)
            if res is not None:
                out = res

        if self.explain:
            explain_msg_kwargs = self.inspector.distribute(
                'explain_message', coll_dict)
            out = self.explain_message(out, **explain_msg_kwargs)

        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        for hook in self._aggregate_forward_pre_hooks.values():
            res = hook(self, (aggr_kwargs, ))
            if res is not None:
                aggr_kwargs = res[0] if isinstance(res, tuple) else res

        out = self.aggregate(out, **aggr_kwargs)

        for hook in self._aggregate_forward_hooks.values():
            res = hook(self, (aggr_kwargs, ), out)
            if res is not None:
                out = res

        update_kwargs = self.inspector.distribute('update', coll_dict)
        out = self.update(out, **update_kwargs)

        ...
        return out
