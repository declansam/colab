import networkx as nx
import pandas as pd
import torch
from explainers.data_util import batched_k_hop_subgraph
from nbfnet.util import remove_duplicate
from nbfnet.tasks import edge_match


def prepare_expl(expl):
    all_query = expl[:, :2]
    explanations = expl[:, 2:]
    unique_query, unique_expl = remove_duplicate(all_query, explanations)
    return unique_query, unique_expl


def load_vocab(vocab_file):
    entity_mapping = {}
    with open(vocab_file, "r") as fin:
        for line in fin:
            k, v = line.strip().split("\t")
            entity_mapping[k] = v

    return entity_mapping


def vizualize_explanation_fb(
    index,
    data,
    id2entity,
    id2relation,
    vocab,
    num_relations,
    output_df,
    expl,
):
    # The GNN_eval Result
    row = output_df.iloc[index]
    if row["Mode"] == 1:
        rel = id2relation[row["Rel"]]
    else:
        rel = id2relation[row["Rel"]] + "_inv"
    head_id = row["Heads"]
    tail_id = row["Tails"]
    rel_id = row["Rel"]
    head = vocab[id2entity[head_id]]
    tail = vocab[id2entity[tail_id]]

    # Print the ranking
    print(f"*** Query: {head, rel}. Answer: {tail} ***")
    print(f"- GNN_eval Rank given the explanation {row['Ranking_GNN_eval']}")
    print(f"- Finetuned Rank given the explanation {row['Ranking_finetune']}")

    nodes, edges = batched_k_hop_subgraph(
        torch.tensor([head_id]).unsqueeze(0), 6, data.edge_index, data.num_nodes
    )

    print(
        f"Is the tail included in the 6-hop neighbor of head in the original graph? {nodes[0, tail_id].item()}"
    )
    full_edge_index = data.edge_index[:, edges.squeeze()]

    if row["Mode"] == 1:
        query = torch.tensor([[head_id, rel_id]])
    else:
        query = torch.tensor([[head_id, rel_id + num_relations // 2]])
    # find the matching explanations
    all_query = expl[0]
    row_id, num_match = edge_match(all_query.T, query.T)
    assert torch.all(num_match == 1)
    expl = expl[1][row_id]

    # expl = explanations[index]
    # query = expl[:2]
    # expl = expl[2:]
    # check if query matches the output.
    # Remove any non-edges:
    non_edge_mask = expl < 0
    expl = expl[~non_edge_mask]
    # Check that the explanation are restricted only to within the ego_network (not necessary)
    # assert torch.all(torch.logical_and(expl, ~edges.squeeze()) == False)
    expl_edge_index = data.edge_index[:, expl]
    expl_edge_type = data.edge_type[expl]

    print(
        f"Is the head included in the Explanation? {torch.any((expl_edge_index == head_id)).item()}"
    )
    print(
        f"Is the tail included in the Explanation? {torch.any((expl_edge_index == tail_id)).item()}"
    )

    # Create a directed graph
    G = nx.MultiDiGraph()

    # Add edges from expl_edge_index
    for i in range(expl_edge_index.shape[1]):
        source = expl_edge_index[0, i].item()
        target = expl_edge_index[1, i].item()
        relation = expl_edge_type[i].item()
        if relation < num_relations // 2:
            rel_name = id2relation[relation]
        else:
            rel_name = id2relation[relation - num_relations // 2] + "_inv"
        # G.add_edge(source, target,arrows='to')
        G.add_edge(source, target, title=rel_name)

    node_ids = torch.unique(expl_edge_index)
    attrs = {}
    for i in node_ids:
        i = i.item()
        attrs[i] = vocab[id2entity[i]]

    nx.set_node_attributes(G, attrs, name="title")
    attrs = {head_id: "#FF0000", tail_id: "#00FF00"}
    nx.set_node_attributes(G, attrs, name="color")
    attrs = {head_id: head, tail_id: tail}
    nx.set_node_attributes(G, attrs, name="title")

    try:
        paths = list(nx.all_simple_paths(G, source=head_id, target=tail_id, cutoff=6))

        attrs = {}
        if len(paths) > 0:
            connected = True
        else:
            connected = False

        print(f"Is the tail connected to the head? {connected}")
        for path in paths:
            src = head_id
            for node in path[1:]:
                edge = (src, node)
                attrs[edge] = {"color": "#b27ebd"}
                src = node

        nx.set_edge_attributes(G, attrs)
    except:
        print(f"Is the tail connected to the head? {False}")

    return G
