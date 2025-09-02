import torch
import json
from tqdm import tqdm
import os.path as osp
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence


if __name__ == "__main__":
    file_names = [
        "/storage/WikiLink/dataset/FB15k-237/train_pooled_nodes.json",
        "/storage/WikiLink/dataset/FB15k-237/valid_pooled_nodes.json",
        "/storage/WikiLink/dataset/FB15k-237/test_pooled_nodes.json",
    ]
    splits = ["train", "valid", "test"]
    save_dir = "/storage/WikiLink/dataset/FB15k-237"

    id2entity, id2relation = torch.load(
        "/storage/ryoji/Graph-Transformer/NBFNet-PyG/misc/fb15k237_id2name.pt"
    )
    entity2id = {}
    for k, v in id2entity.items():
        entity2id[v] = k
    rel2id = {}
    for k, v in id2relation.items():
        rel2id[v] = k
    num_relations = len(rel2id)

    all_subgraphs = defaultdict(list)

    for file_name, split in zip(file_names, splits):
        count = 0
        included = 0

        with open(file_name) as f:
            subgraphs = json.load(f)

        for subgraph in tqdm(subgraphs, desc=f"Processing {split}"):
            for mode in ["tail", "head"]:
                head = subgraph["triple"]["head"]
                rel = subgraph["triple"]["relation"]
                tail = subgraph["triple"]["tail"]
                if mode == "tail":
                    head_id = entity2id[head]
                    rel_id = rel2id[rel]
                    tail_id = entity2id[tail]
                else:
                    head_id = entity2id[tail]
                    rel_id = rel2id[rel] + num_relations
                    tail_id = entity2id[head]

                graph = [head_id, rel_id]
                nodes = []
                for entity in subgraph["pooled_nodes"][mode]:
                    nodes.append(entity2id[entity])

                if tail_id in nodes:
                    included += 1
                graph.extend(nodes)
                all_subgraphs[split].append(torch.tensor(graph, dtype=torch.int32))
                count += 1

        print(f"Inclusion Rate for {split}: {included/count}")

    for split in splits:
        subgraphs = all_subgraphs[split]
        subgraphs = pad_sequence(subgraphs, padding_value=-1, batch_first=True)
        torch.save(subgraphs, osp.join(save_dir, f"{split}_explanations.pt"))
