# %%
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import pandas as pd
from notebooks.vizualize import vizualize_explanation_fb, load_vocab, prepare_expl
import networkx as nx
from pyvis.network import Network

# %%
dataset = torch.load(
    "/storage/ryoji/Graph-Transformer/NBFNet-PyG/misc/fb15k237_dataset.pt"
)
id2entity, id2relation = torch.load(
    "/storage/ryoji/Graph-Transformer/NBFNet-PyG/misc/fb15k237_id2name.pt"
)
vocab_file = "/storage/ryoji/Graph-Transformer/NBFNet-PyG/data/fb15k237_entity.txt"
vocab = load_vocab(vocab_file)

# %%
# Configurable
split = "test"  # the split
ratio = 25  # the top r ratio of edges
root_dir = "/storage/ryoji/Graph-Transformer/NBFNet-PyG/explanation/EdgeRandomWalk/NBFNet/FB15k-237/ERW"
# root_dir = "/storage/ryoji/Graph-Transformer/NBFNet-PyG/explanation/EdgeRandomWalk/NBFNet/FB15k-237/ERW_inv"

# %%
if split == "valid":
    data_index = 1
if split == "test":
    data_index = 2

# The path of dataset
output_file = f"{root_dir}/{split}_output_factual_eval_hard_edge_mask_top_k_{ratio}.pt"
explanation_file = (
    f"{root_dir}/{split}_explanations_factual_eval_hard_edge_mask_top_k_{ratio}.pt"
)
finetune_output_file = (
    f"{root_dir}/run_NBFNet_hard_edge_mask_top_k_{ratio}/{split}_output.pt"
)

# Load the dataset
explanations = torch.load(explanation_file, map_location="cpu")
outputs = torch.load(output_file, map_location="cpu")
finetune_outputs = torch.load(finetune_output_file, map_location="cpu")
output_df = pd.DataFrame(outputs)
finetune_df = pd.DataFrame(finetune_outputs)
data = dataset[data_index]

# Drop any duplicate records, this happens from DDP, DistributedSampler drop_last = False.
output_df = output_df.drop_duplicates(subset=["Heads", "Tails", "Rel", "Mode"])
finetune_df = finetune_df.drop_duplicates(subset=["Heads", "Tails", "Rel", "Mode"])

# Merge the output and finetune dataframe into one
output_df = pd.merge(
    output_df,
    finetune_df,
    on=["Heads", "Tails", "Rel", "Mode"],
    suffixes=("_GNN_eval", "_finetune"),
    how="left",
)

# Preprocess the explanations
expl = prepare_expl(explanations)

# %%
# Control the index (row number of output_df of the instance you want to inspect here)
index = 4

# %%
G = vizualize_explanation_fb(
    index,
    data,
    id2entity,
    id2relation,
    vocab,
    dataset.num_relations,
    output_df,
    expl,
)

# %%
# *** Explanation Graph Visualization ***
# RED NODE: The Query Head (if inside the explanation)
# GREEN NODE: The Answer Tail (if inside the explanation)
# Purple Edges: Edges that are in any path that connected Red to Green
net = Network(
    notebook=True,
    cdn_resources="remote",
    bgcolor="#222222",
    font_color="white",
    height="750px",
    width="100%",
    select_menu=True,
    filter_menu=True,
    directed=True,
)
net.from_nx(G)
net.inherit_edge_colors(False)
net.set_edge_smooth("dynamic")
net.show("graph.html")

# %%
