import torch
from itertools import count
from heapq import heappop, heappush

try:
    from explainers.data_util import create_batched_data
except:
    from explainers.data_util import create_batched_data
import copy
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import time
import concurrent.futures

"""
Path finding utils
"""


def multigraph_to_graph(data, mask, neg_path_score):
    """
    Knowledge graph can be multigraph, convert this to digraph (only one edge in u, v) by selecting the relation with highest weight.
    (we select the relation with highest weight by selecting the relation of lowest neg weight!)
    This is necessary as Dijkstra's algorithm is not implemented for multigraph & we don't care about the relation with lower weight.
    We only select the relation with highest weight because the path finding algorithm will try to find the path with highest score.
    """
    # each edge in every subgraph gets a unique id
    offset = data.edge_batch * data.num_nodes
    unique_edges = data.edge_index + offset

    # for each original edge, inverse_indices is the corresponding index in unique edges
    unique_edges, inverse_indices = torch.unique(
        unique_edges, return_inverse=True, dim=1
    )

    num_relations = torch.max(data.edge_type) + 1
    # for each unique edge, multi_edge_weights will get the score of the mask for each available edge type
    multi_edge_weights = torch.zeros(
        (unique_edges.size(1), num_relations), device=unique_edges.device
    ).fill_(float("inf"))
    multi_edge_weights[inverse_indices, data.edge_type] = neg_path_score
    # for each unique edge, take the relation with highest weight (lowest negative weight)
    neg_path_score, edge_type = torch.min(multi_edge_weights, dim=-1)

    # for the relations with highest weight, get their corresponding edge mask
    multi_edge_mask = torch.zeros(
        (unique_edges.size(1), num_relations), device=unique_edges.device
    ).fill_(float("inf"))
    multi_edge_mask[inverse_indices, data.edge_type] = mask
    path_mask = multi_edge_mask[torch.arange(unique_edges.size(1)), edge_type]
    assert ~torch.any(path_mask == float("inf"))

    # we also need to keep the edge mask that were not selected!
    orig_edge_id = torch.zeros(
        (unique_edges.size(1), num_relations),
        device=unique_edges.device,
        dtype=torch.long,
    )
    # for each unique edges and their rels, get the original edge id
    orig_edge_id[inverse_indices, data.edge_type] = torch.arange(
        data.edge_type.size(0), device=unique_edges.device
    )
    # get the mask for the unselected edges
    unselected_edge_mask = torch.zeros_like(orig_edge_id, dtype=torch.bool)
    unselected_edge_mask[inverse_indices, data.edge_type] = True
    unselected_edge_mask[torch.arange(unique_edges.size(1)), edge_type] = (
        False  # mark the selected ones as false
    )
    unselected_edge_id = orig_edge_id[unselected_edge_mask]
    # the addition of selected and unselected should be equal to everything!
    assert path_mask.size(0) + unselected_edge_id.size(0) == mask.size(0)

    # map the unique edges back to its original
    # get the edge_batch for the unique edges
    batch_size = data.edge_filter.size(0)
    unique_edge_batch = torch.zeros(
        (unique_edges.size(1), batch_size), device=unique_edges.device, dtype=torch.long
    )
    unique_edge_batch[inverse_indices, data.edge_batch] = 1
    assert torch.all(
        unique_edge_batch.sum(dim=-1) == 1
    )  # there should only be one batch id for each unique edge
    edge_batch = unique_edge_batch.nonzero().T[1]
    rev_offset = edge_batch * data.num_nodes
    edge_index = unique_edges - rev_offset

    # digraph used specifically for path finding
    path_data = copy.copy(data)
    path_data.edge_index = edge_index
    path_data.edge_type = edge_type
    path_data.edge_batch = edge_batch
    index, counts = torch.unique(
        path_data.edge_batch, return_counts=True
    )  # ensure if no edge the num_edge will be 0
    subgraph_num_edges = torch.zeros_like(path_data.subgraph_num_edges)
    subgraph_num_edges[index] = counts
    path_data.subgraph_num_edges = subgraph_num_edges
    path_data = create_batched_data(path_data)

    return path_data, path_mask, neg_path_score, unselected_edge_id


def bidirectional_dijkstra(
    edge_index, src_nid, tgt_nid, weight, ignore_nodes=None, ignore_edges=None
):
    """Dijkstra's algorithm for shortest paths using bidirectional search.

    Adapted from NetworkX _bidirectional_dijkstra
    https://networkx.org/documentation/stable/_modules/networkx/algorithms/simple_paths.html

    Parameters
    ----------
    edge_index: the edge index

    src_nid : int
        source node id

    tgt_nid : int
        target node id

    weight: callable function,
       Takes in two node ids and a key and return the edge weight/edge id.

    ignore_nodes : container of nodes
       nodes to ignore, optional

    ignore_edges : container of edges
       edges to ignore, optional

    Returns
    -------
    distance, [paths_nodes, paths_edge_id]
    This returns the shortest path with its associated distance & paths by their nodes & edge_id in edge_index

    """
    if src_nid == tgt_nid:  # self loops
        try:
            w = weight(src_nid, tgt_nid, "weight")
            edge_id = weight(src_nid, tgt_nid, "edge_id")
            return (w, [[src_nid], [edge_id]])
        except KeyError:
            return (0, [[src_nid], []])

    src, tgt = edge_index
    Gpred = lambda i: src[tgt == i].tolist()
    Gsucc = lambda i: tgt[src == i].tolist()

    if ignore_nodes:

        def filter_iter(nodes):
            def iterate(v):
                for w in nodes(v):
                    if w not in ignore_nodes:
                        yield w

            return iterate

        Gpred = filter_iter(Gpred)
        Gsucc = filter_iter(Gsucc)

    if ignore_edges:

        def filter_pred_iter(pred_iter):
            def iterate(v):
                for w in pred_iter(v):
                    if (w, v) not in ignore_edges:
                        yield w

            return iterate

        def filter_succ_iter(succ_iter):
            def iterate(v):
                for w in succ_iter(v):
                    if (v, w) not in ignore_edges:
                        yield w

            return iterate

        Gpred = filter_pred_iter(Gpred)
        Gsucc = filter_succ_iter(Gsucc)

    push = heappush
    pop = heappop
    # Init:   Forward             Backward
    dists = [{}, {}]  # dictionary of final distances
    paths = [
        {src_nid: [[src_nid], []]},
        {tgt_nid: [[tgt_nid], []]},
    ]  # dictionary of paths tuple of (nodes_in_path, edge_id_of_path)
    fringe = [[], []]  # heap of (distance, node) tuples for
    # extracting next node to expand
    seen = [{src_nid: 0}, {tgt_nid: 0}]  # dictionary of distances to
    # nodes seen
    c = count()
    # initialize fringe heap
    push(fringe[0], (0, next(c), src_nid))
    push(fringe[1], (0, next(c), tgt_nid))
    # neighs for extracting correct neighbor information
    neighs = [Gsucc, Gpred]
    # variables to hold shortest discovered path
    # finaldist = 1e30000
    finalpath = []
    dir = 1

    while fringe[0] and fringe[1]:
        # choose direction
        # dir == 0 is forward direction and dir == 1 is back
        dir = 1 - dir
        # extract closest to expand
        (dist, _, v) = pop(fringe[dir])
        if v in dists[dir]:
            # Shortest path to v has already been found
            continue
        # update distance
        dists[dir][v] = dist  # equal to seen[dir][v]
        if v in dists[1 - dir]:
            # if we have scanned v in both directions we are done
            # we have now discovered the shortest path
            return (finaldist, finalpath)

        for w in neighs[dir](v):
            if dir == 0:  # forward
                minweight = weight(v, w, "weight")
                edge_id = weight(v, w, "edge_id")
                vwLength = dists[dir][v] + minweight
            else:  # back, must remember to change v,w->w,v
                minweight = weight(w, v, "weight")
                edge_id = weight(w, v, "edge_id")
                vwLength = dists[dir][v] + minweight

            if w in dists[dir]:
                if vwLength < dists[dir][w]:
                    raise ValueError("Contradictory paths found: negative weights?")
            elif w not in seen[dir] or vwLength < seen[dir][w]:
                # relaxing
                seen[dir][w] = vwLength
                push(fringe[dir], (vwLength, next(c), w))
                paths[dir][w] = []
                paths[dir][w].append(paths[dir][v][0] + [w])
                paths[dir][w].append(paths[dir][v][1] + [edge_id])
                if w in seen[0] and w in seen[1]:
                    # see if this path is better than the already
                    # discovered shortest path
                    totaldist = seen[0][w] + seen[1][w]
                    if finalpath == [] or finaldist > totaldist:
                        finaldist = totaldist
                        revpath_nodes = paths[1][w][0][:]
                        revpath_nodes.reverse()
                        revpath_edge_id = paths[1][w][1][:]
                        revpath_edge_id.reverse()
                        finalpath = [
                            paths[0][w][0] + revpath_nodes[1:],
                            paths[0][w][1] + revpath_edge_id,
                        ]
    raise ValueError("No paths found")


class PathBuffer:
    """For shortest paths finding

    Adapted from NetworkX shortest_simple_paths
    https://networkx.org/documentation/stable/_modules/networkx/algorithms/simple_paths.html

    """

    def __init__(self):
        self.paths = set()
        self.sortedpaths = list()
        self.counter = count()

    def __len__(self):
        return len(self.sortedpaths)

    def push(self, cost, path):
        hashable_path = tuple(path[0])  # path hash by their node ids
        if hashable_path not in self.paths:
            heappush(self.sortedpaths, (cost, next(self.counter), path))
            self.paths.add(hashable_path)

    def pop(self):
        (cost, num, path) = heappop(self.sortedpaths)
        hashable_path = tuple(path[0])
        self.paths.remove(hashable_path)
        return path


def shortest_paths_generator_inner(
    edge_index,
    src_nid,
    tgt_nid,
    weight,
    listA,
    listB,
    prev_path,
    length_func,
    ignore_nodes_init=None,
    ignore_edges_init=None,
):
    if not prev_path:
        length, path = bidirectional_dijkstra(
            edge_index, src_nid, tgt_nid, weight, ignore_nodes_init, ignore_edges_init
        )
        listB.push(length, path)
    else:
        ignore_nodes = set(ignore_nodes_init) if ignore_nodes_init else set()
        ignore_edges = set(ignore_edges_init) if ignore_edges_init else set()
        for i in range(1, len(prev_path)):
            root = prev_path[0][:i]
            edge_id_root = prev_path[1][:i]
            root_length = length_func(root)
            for path in listA:
                if path[0][:i] == root:
                    ignore_edges.add((path[0][i - 1], path[0][i]))
            try:
                length, spur = bidirectional_dijkstra(
                    edge_index,
                    root[-1],
                    tgt_nid,
                    ignore_nodes=ignore_nodes,
                    ignore_edges=ignore_edges,
                    weight=weight,
                )
                path = []
                path.append(root[:-1] + spur[0])
                path.append(edge_id_root[:-1] + spur[1])
                listB.push(root_length + length, path)
            except ValueError:
                pass
            ignore_nodes.add(root[-1])


def shortest_paths_generator_with_budget(
    edge_index,
    src_nid,
    tgt_nid,
    weight,
    eval_mask_type,
    budget,
    timeout_duration,
    ignore_nodes_init=None,
    ignore_edges_init=None,
):

    def length_func(path):
        return sum(weight(u, v, "weight") for (u, v) in zip(path, path[1:]))

    listA = list()
    listB = PathBuffer()
    prev_path = None
    satisfied_budget = False
    selected_nodes = set()
    selected_edges = set()
    start_time = time.time()
    while not satisfied_budget:
        if time.time() - start_time > timeout_duration:
            break
        shortest_paths_generator_inner(
            edge_index,
            src_nid,
            tgt_nid,
            weight,
            listA,
            listB,
            prev_path,
            length_func,
            ignore_nodes_init,
            ignore_edges_init,
        )
        if listB:
            path = listB.pop()
            nodes = set(path[0])
            edge_ids = set(path[1])
            node_union = selected_nodes.union(nodes)
            edge_union = selected_edges.union(edge_ids)
            if "node" in eval_mask_type and len(node_union) > budget:
                # we satisfied the budget!
                satisfied_budget = True
                break
            elif "edge" in eval_mask_type and len(edge_union) > budget:
                # we satisfied the budget!
                satisfied_budget = True
                break
            yield path
            if len(path[0]) == 1:  # if self loop, break from it
                break
            listA.append(path)
            prev_path = path
            # update the nodes and edges
            selected_nodes = node_union
            selected_edges = edge_union
        else:
            break


def k_shortest_paths_generator(
    edge_index,
    src_nid,
    tgt_nid,
    weight,
    k=5,
    timeout_duration=None,
    ignore_nodes_init=None,
    ignore_edges_init=None,
):
    """Generate at most `k` simple paths in the graph g from src_nid to tgt_nid,
       each with maximum lenghth `max_length`, return starting from the shortest ones.
       If a weighted shortest path search is to be used, no negative weights are allowed.

    Adapted from NetworkX shortest_simple_paths
    https://networkx.org/documentation/stable/_modules/networkx/algorithms/simple_paths.html

    Parameters
    ----------
    g : dgl graph

    src_nid : int
        source node id

    tgt_nid : int
        target node id

    weight: callable function, optional
       Takes in two node ids and return the edge weight.

    k: int
       number of paths

    ignore_nodes_init : set of nodes
       nodes to ignore, optional

    ignore_edges_init : set of edges
       edges to ignore, optional

    Returns
    -------
    path_generator: generator
       A generator that produces lists of tuples (path score, path), in order from
       shortest to longest. Each path is a list of node ids

    """

    def length_func(path):
        return sum(weight(u, v, "weight") for (u, v) in zip(path, path[1:]))

    listA = list()
    listB = PathBuffer()
    prev_path = None
    start_time = time.time()
    while not prev_path or len(listA) < k:
        if time.time() - start_time > timeout_duration:
            break
        shortest_paths_generator_inner(
            edge_index,
            src_nid,
            tgt_nid,
            weight,
            listA,
            listB,
            prev_path,
            length_func,
            ignore_nodes_init,
            ignore_edges_init,
        )
        if listB:
            path = listB.pop()
            yield path
            if len(path[0]) == 1:  # if self loop, break from it
                break
            listA.append(path)
            prev_path = path
        else:
            break


def k_shortest_paths_with_max_length(
    edge_index,
    src_nid,
    tgt_nid,
    weight,
    k=5,
    max_length=None,
    timeout_duration=None,
    ignore_nodes=None,
    ignore_edges=None,
):
    """Generate at most `k` simple paths in the graph g from src_nid to tgt_nid,
       each with maximum lenghth `max_length`, return starting from the shortest ones.
       If a weighted shortest path search is to be used, no negative weights are allowed.

    Parameters
    ----------
       See function `k_shortest_paths_generator`

    Return
    -------
    paths: list of lists
       Each list is a path containing node ids
    """
    path_generator = k_shortest_paths_generator(
        edge_index,
        src_nid,
        tgt_nid,
        weight=weight,
        k=k,
        timeout_duration=timeout_duration,
        ignore_nodes_init=ignore_nodes,
        ignore_edges_init=ignore_edges,
    )

    try:
        if max_length:
            paths = [path for path in path_generator if len(path[0]) <= max_length + 1]
        else:
            paths = list(path_generator)

    except ValueError:
        paths = [[]]

    return paths


def find_paths(edge_index, weight, k, max_length, src_nid, tgt_nid):

    paths = k_shortest_paths_with_max_length(
        edge_index,
        src_nid,
        tgt_nid,
        weight=lambda u, v, key: weight[(u, v)][key],
        k=k,
        max_length=max_length,
    )
    edges_in_path = torch.zeros(edge_index.size(1), dtype=torch.bool)
    eids = []
    for path in paths:
        eid = path[1]  # get the edge ids of the paths
        eids.extend(eid)  # extend the list of edge ids
    edges_in_path[eids] = True
    return edges_in_path


def parallel_path_finder(
    data,
    weight,
    k=5,
    max_length=None,
    num_workers=1,
    ignore_nodes=None,
    ignore_edges=None,
):
    # offset it so each subgraph gets unique edge index
    batch_size = data.central_node_index.size(0)
    offsets = torch.arange(batch_size) * data.num_nodes
    central_node_index = data.central_node_index.to("cpu") + offsets.unsqueeze(1)

    edge_index = data.edge_batch * data.num_nodes + data.edge_index
    edge_index = edge_index.to("cpu")
    weight = weight.detach().to("cpu")

    u, v = edge_index
    weight_and_edge_id_map = {
        edge: {"weight": weight[i].item(), "edge_id": i}
        for i, edge in enumerate(zip(u.tolist(), v.tolist()))
    }

    def weight_and_edge_id_func(u, v, key):
        return weight_and_edge_id_map[(u, v)][key]

    heads = central_node_index[:, 0]
    tails = central_node_index[:, 1:]

    heads = heads.repeat_interleave(tails.size(1)).tolist()
    tails = tails.flatten().tolist()

    edge_index = edge_index.unsqueeze(0).repeat(len(heads), 1, 1)
    weight = [weight_and_edge_id_map] * len(heads)
    k = [k] * len(heads)
    max_length = [max_length] * len(heads)

    edges_in_path = torch.zeros(data.edge_batch.size(0), dtype=torch.bool)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in executor.map(
            find_paths, edge_index, weight, k, max_length, heads, tails, chunksize=10
        ):
            edges_in_path |= result

    return edges_in_path


def sequential_path_finder(
    data,
    weight,
    k=5,
    max_length=None,
    return_paths=False,
    timeout_duration=None,
    ignore_nodes=None,
    ignore_edges=None,
):
    # offset it so each subgraph gets unique edge index
    batch_size = data.central_node_index.size(0)
    offsets = torch.arange(batch_size) * data.num_nodes
    central_node_index = data.central_node_index.to("cpu") + offsets.unsqueeze(1)

    edge_index = data.edge_batch * data.num_nodes + data.edge_index
    edge_index = edge_index.to("cpu")
    weight = weight.detach().to("cpu")

    u, v = edge_index
    weight_and_edge_id_map = {
        edge: {"weight": weight[i].item(), "edge_id": i}
        for i, edge in enumerate(zip(u.tolist(), v.tolist()))
    }

    def weight_and_edge_id_func(u, v, key):
        return weight_and_edge_id_map[(u, v)][key]

    heads = central_node_index[:, 0]
    tails = central_node_index[:, 1:]

    # Combine results from all workers
    edges_in_path = torch.zeros(edge_index.size(1), dtype=torch.bool)

    path_collection = defaultdict(list)

    for batch_id in range(batch_size):
        for tgt_node in range(tails.size(1)):
            src_nid, tgt_nid = heads[batch_id].item(), tails[batch_id][tgt_node].item()
            paths = k_shortest_paths_with_max_length(
                edge_index,
                src_nid,
                tgt_nid,
                weight_and_edge_id_func,
                k,
                max_length,
                timeout_duration,
            )
            eids = []
            for path in paths:
                if len(path) == 0:
                    continue
                # the nodes in the path are labeled in unique index, relabel them back to original
                nid = path[0]  # get the node ids of the paths
                relabeled_nids = torch.tensor(nid) - batch_id * data.num_nodes
                eid = path[1]  # get the edge ids of the paths
                eids.extend(eid)  # extend the list of edge ids
                path_collection[batch_id].append([relabeled_nids.tolist(), eid])
            edges_in_path[eids] = True  # mark these edges as being used in the paths

    if return_paths:
        return path_collection
    return edges_in_path


def sequential_path_finder_until_budget(
    data,
    weight,
    max_length=None,
    return_paths=False,
    eval_mask_type=None,
    max_budget=None,
    timeout_duration=None,
    ignore_nodes=None,
    ignore_edges=None,
):
    """
    Iterate until you either meet the max budget or OOT.
    """
    assert max_budget > 0
    # offset it so each subgraph gets unique edge index
    batch_size = data.central_node_index.size(0)
    offsets = torch.arange(batch_size) * data.num_nodes
    central_node_index = data.central_node_index.to("cpu") + offsets.unsqueeze(1)

    edge_index = data.edge_batch * data.num_nodes + data.edge_index
    edge_index = edge_index.to("cpu")
    weight = weight.detach().to("cpu")

    u, v = edge_index
    weight_and_edge_id_map = {
        edge: {"weight": weight[i].item(), "edge_id": i}
        for i, edge in enumerate(zip(u.tolist(), v.tolist()))
    }

    def weight_and_edge_id_func(u, v, key):
        return weight_and_edge_id_map[(u, v)][key]

    heads = central_node_index[:, 0]
    tails = central_node_index[:, 1:]

    # Combine results from all workers
    edges_in_path = torch.zeros(edge_index.size(1), dtype=torch.bool)

    path_collection = defaultdict(list)

    for batch_id in range(batch_size):

        for tgt_node in range(tails.size(1)):
            src_nid, tgt_nid = heads[batch_id].item(), tails[batch_id][tgt_node].item()

            path_generator = shortest_paths_generator_with_budget(
                edge_index,
                src_nid,
                tgt_nid,
                weight_and_edge_id_func,
                eval_mask_type,
                max_budget,
                timeout_duration,
                ignore_nodes,
                ignore_edges,
            )
            try:
                if max_length:
                    paths = [
                        path
                        for path in path_generator
                        if len(path[0]) <= max_length + 1
                    ]
                else:
                    paths = list(path_generator)
            except ValueError:
                paths = [[]]

            eids = []
            for path in paths:
                if len(path) == 0:
                    continue
                # the nodes in the path are labeled in unique index, relabel them back to original
                nid = path[0]  # get the node ids of the paths
                relabeled_nids = torch.tensor(nid) - batch_id * data.num_nodes
                eid = path[1]  # get the edge ids of the paths
                eids.extend(eid)  # extend the list of edge ids
                path_collection[batch_id].append([relabeled_nids.tolist(), eid])
            edges_in_path[eids] = True  # mark these edges as being used in the paths

    if return_paths:
        return path_collection
    return edges_in_path


# import time
# import networkx as nx
# import nx_parallel as nxp
# from torch_geometric.utils import to_networkx


# if __name__ == '__main__':
#     # test parallel path finder algorithm
#     data, weight, k, max_length = torch.load('/storage/ryoji/Graph-Transformer/NBFNet-PyG/parallel_path.pt')
#     start = time.time()
#     edges_in_path_sq = sequential_path_finder(data, weight, k, max_length)
#     end = time.time()
#     print(f"* Getting paths took {(end-start):.2f}s with 1 worker*")
#     data.weight = weight
#     raise Exception()
