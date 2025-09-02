import torch
import pandas as pd
from utils import batched_k_hop_subgraph
import networkx as nx
from tqdm import tqdm
import argparse

def inspect_explanation(index):
    row = output_df.iloc[index]
    if row['Mode'] == 1:
        rel = id2relation[row['Rel']]
    else:
        rel = id2relation[row['Rel']]+'_inv'

    head_id = row['Heads']
    tail_id = row['Tails']
    head = id2entity[head_id]
    tail = id2entity[tail_id]

    # print(f"*** Query: {head, rel}. Answer: {tail}, Rank given the explanation: {row['Ranking']} ***")
    nodes, edges = batched_k_hop_subgraph(torch.tensor([head_id]), 6, data.edge_index, data.num_nodes)

    tail_included_in_ego = nodes[0, tail_id].item()
    num_nodes_ego = nodes.sum().item()
    num_edges_ego = edges.sum().item()

    expl = explanations[index]
    # Check that the explanation are restricted only to within the ego_network
    assert torch.all(torch.logical_and(expl, ~edges.squeeze()) == False)
    expl_edge_index = data.edge_index[:, expl.to(torch.bool)]
    num_nodes_expl = expl_edge_index.unique().size(0)
    num_edges_expl = expl_edge_index.size(1)

    assert num_nodes_expl <= num_nodes_ego and num_edges_expl <= num_edges_ego

    # print(f'Is the head included in the Explanation? {torch.any((expl_edge_index == head_id)).item()}')
    head_included_in_expl = torch.any((expl_edge_index == head_id)).item()
    tail_included_in_expl = torch.any((expl_edge_index == tail_id)).item()

    if head_included_in_expl and tail_included_in_expl:
        nodes, edges = batched_k_hop_subgraph(torch.tensor([tail_id]), 6, expl_edge_index, data.num_nodes)
        tail_connected_to_head_in_expl = nodes[0, head_id].item()
    else:
        tail_connected_to_head_in_expl = False
    
    return tail_included_in_ego, head_included_in_expl, tail_included_in_expl, tail_connected_to_head_in_expl, num_nodes_ego, num_edges_ego, num_nodes_expl, num_edges_expl


if __name__ == '__main__':
    dataset = torch.load('/storage/ryoji/Graph-Transformer/NBFNet-PyG/wn18rr_dataset.pt')
    id2entity, id2relation = torch.load('/storage/ryoji/Graph-Transformer/NBFNet-PyG/wn18rr_id2name.pt')

    split = 'valid' # the split
    ratio = 0.3 # the top r ratio of edges

    if split == 'valid':
        data_index = 1
    if split == 'test':
        data_index = 2

    output_file = f'/storage/ryoji/Graph-Transformer/NBFNet-PyG/explanation/NBFNet/WN18RR/explanation_output/{split}_output_hard_edge_mask_top_ratio_{ratio}.pt'
    explanation_file = f'/storage/ryoji/Graph-Transformer/NBFNet-PyG/explanation/NBFNet/WN18RR/explanation_output/{split}_explanations_hard_edge_mask_top_ratio_{ratio}.pt'

    explanations = torch.load(explanation_file)
    outputs = torch.load(output_file)
    output_df = pd.DataFrame(outputs)

    data = dataset[data_index]

    
    for index in tqdm(range(len(output_df.index))):
        tail_included_in_ego, head_included_in_expl, tail_included_in_expl, tail_connected_to_head_in_expl, num_nodes_ego, num_edges_ego, num_nodes_expl, num_edges_expl = inspect_explanation(index)
        # Add new variables to the corresponding row in output_df
        output_df.at[index, 'tail_included_in_ego'] = tail_included_in_ego
        output_df.at[index, 'head_included_in_expl'] = head_included_in_expl
        output_df.at[index, 'tail_included_in_expl'] = tail_included_in_expl
        output_df.at[index, 'tail_connected_to_head_in_expl'] = tail_connected_to_head_in_expl
        output_df.at[index, 'num_nodes_ego'] = num_nodes_ego
        output_df.at[index, 'num_edges_ego'] = num_edges_ego
        output_df.at[index, 'num_nodes_expl'] = num_nodes_expl
        output_df.at[index, 'num_edges_expl'] = num_edges_expl

    torch.save(output_df,f'/storage/ryoji/Graph-Transformer/NBFNet-PyG/explanation/NBFNet/WN18RR/explanation_output/{split}_inspect_hard_edge_mask_top_ratio_{ratio}.pt')

