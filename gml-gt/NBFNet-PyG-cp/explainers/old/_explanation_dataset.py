from tqdm import tqdm
import os
import pdb
import torch
import numpy as np
from itertools import chain
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import k_hop_subgraph
from explainers._util import remove_target_edge


class ExplanationDataset(InMemoryDataset):
    def __init__(
        self, root, name, dataset, split, hops, transform=None, pre_transform=None
    ):
        self.folder = os.path.join(root, name + "_explanation")
        self.dataset = dataset
        self.filtered_data = Data(
            edge_index=dataset.target_edge_index, edge_type=dataset.target_edge_type
        )
        self.hops = hops
        self.split = split

        super().__init__(self.folder, transform, pre_transform)
        # (self.data, self.slices), self.train_mask, self.valid_mask, self.test_mask = torch.load(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.id2entity = self.dataset.id2entity
        self.id2relation = self.dataset.id2relation
        self.num_relations = self.dataset.num_relations
        self.dataset = None
        # For some reason, these tensors needs to be cloned for multiprocessing to work.
        # It might be related to how view cannot be accessed for multuprocessing. https://github.com/pyg-team/pytorch_geometric/discussions/6919
        # self.train_mask = self.train_mask.clone()
        # self.valid_mask = self.valid_mask.clone()
        # self.test_mask = self.test_mask.clone()

    @property
    def raw_file_names(self):
        return "raw.txt"

    @property
    def processed_file_names(self):
        # return 'data_subsample.pt'
        return f"{self.split}.pt"
        # return 'data.pt'

    def download(self):
        pass

    def process(self):
        # load the original datasets
        split_masks = []
        data_list = []

        for data, split in zip(self.dataset, ["train", "valid", "test"]):
            if split != self.split:
                continue
            count = 0
            test_triplets = torch.cat(
                [data.target_edge_index, data.target_edge_type.unsqueeze(0)]
            ).t()
            for i in tqdm(
                range(test_triplets.shape[0]),
                desc=f"Obtaining Explanation Dataset for {split}",
            ):
                triplet = test_triplets[i]
                # mask the target edge
                if split == "train":
                    masked_data = remove_target_edge(
                        self.dataset.num_relations // 2,
                        data,
                        triplet[0].unsqueeze(0),
                        triplet[1].unsqueeze(0),
                        triplet[2].unsqueeze(0),
                    )
                else:
                    masked_data = data

                for head, mode in zip(triplet[:-1], ["tail-batch", "head-batch"]):
                    head = head.unsqueeze(0)
                    # get the k-hop neighborhood of the graph centered around the head node, relabeled
                    if head in masked_data.edge_index:
                        selected_nodes, edge_index, mapped_head, edge_mask = (
                            k_hop_subgraph(
                                head,
                                self.hops,
                                masked_data.edge_index,
                                relabel_nodes=True,
                            )
                        )
                    else:  # sometimes the node cannot be found in the edge
                        selected_nodes = head
                        edge_index = torch.tensor([[], []], dtype=torch.long)
                        mapped_head = torch.tensor([0])
                        edge_mask = torch.zeros(
                            masked_data.edge_type.shape, dtype=torch.bool
                        )

                    s_data = Data(
                        x=selected_nodes.unsqueeze(0).T,
                        edge_index=edge_index,
                        edge_type=masked_data.edge_type[edge_mask],
                        split=masked_data.split,
                        num_nodes_subgraph=selected_nodes.shape[0],
                        # node_id = torch.arange(selected_nodes.shape[0]),
                        num_edges_subgraph=edge_index.shape[1],
                        full_num_nodes=data.num_nodes,
                    )

                    s_data.center_node_index = mapped_head

                    if mode == "tail-batch":
                        center_node, positive_arg, eval_rel = triplet

                        arg = 0  # 0 means we will later look at edges that has the node in the source when creating filter
                        candidate_arg = 1  # 1 means we will later look at the candidates in the target
                        s_data.mode = 1  # 1 for tail batch

                    elif mode == "head-batch":  # fix rel, tail
                        positive_arg, center_node, eval_rel = triplet

                        arg = 1  # 1 means we will later look at edges that has the node in the target when creating filter
                        candidate_arg = 0  # 0 means we will later look at the candidates in the source
                        s_data.mode = 0  # 0 for head batch

                    s_data.positive_arg = positive_arg.unsqueeze(0)
                    s_data.eval_rel = eval_rel

                    if positive_arg in selected_nodes:
                        s_data.true_candidate_in_graph = True
                        s_data.pos_arg_index = torch.bucketize(
                            positive_arg, selected_nodes, right=False
                        ).unsqueeze(0)
                    else:
                        s_data.true_candidate_in_graph = False
                        s_data.pos_arg_index = torch.tensor([0])

                    # data used for loss
                    center_node_mask = (
                        data.edge_index[arg] == center_node
                    )  # find the edges in the split that has the center node in source/target
                    rel_mask = (
                        data.edge_type == eval_rel
                    )  # find the triples with matching relations
                    candidate_node_mask = torch.isin(
                        data.edge_index[candidate_arg], selected_nodes
                    )
                    candidate_mask = torch.logical_and(
                        torch.logical_and(center_node_mask, rel_mask),
                        candidate_node_mask,
                    )
                    candidates = data.edge_index[candidate_arg][candidate_mask]
                    mapped_candidates = torch.bucketize(
                        candidates, selected_nodes, right=False
                    )
                    s_data.label_node_index = mapped_candidates

                    # data used for ranking
                    center_node_mask = (
                        self.filtered_data.edge_index[arg] == center_node
                    )  # find the edges in all the data that has the center node in source/target
                    rel_mask = (
                        self.filtered_data.edge_type == eval_rel
                    )  # find the triples with matching relations
                    candidate_node_mask = torch.isin(
                        self.filtered_data.edge_index[candidate_arg], selected_nodes
                    )
                    candidate_mask = torch.logical_and(
                        torch.logical_and(center_node_mask, rel_mask),
                        candidate_node_mask,
                    )
                    candidates = self.filtered_data.edge_index[candidate_arg][
                        candidate_mask
                    ]
                    mapped_candidates = torch.bucketize(
                        candidates, selected_nodes, right=False
                    )  # these are mapped indices of candidates that has the same query

                    if (
                        positive_arg in selected_nodes
                    ):  # exclude the positive arg from the filtered nodes
                        node_eval_filter = mapped_candidates != s_data.pos_arg_index
                        s_data.eval_filter_index = mapped_candidates[node_eval_filter]
                    else:
                        s_data.eval_filter_index = mapped_candidates

                    data_list.append(s_data)

                    # if split == 'train':
                    #     split_masks.append([1, 0, 0])
                    # elif split == 'valid':
                    #     split_masks.append([0, 1, 0])
                    # else:
                    #     split_masks.append([0, 0, 1])

                count += 1
                # if count >= 100:
                #     break

        split_masks = torch.tensor(split_masks, dtype=torch.bool)
        # torch.save((self.collate(data_list), split_masks[:, 0].clone(), split_masks[:, 1].clone(), split_masks[:, 2].clone()), self.processed_paths[0])
        torch.save(self.collate(data_list), self.processed_paths[0])


from torch.utils.data import Dataset


class DynamicExplanationDataset(
    Dataset
):  # Use torch dataset since we don't want to update the indices
    def __init__(self, root, name, dataset, split, hops):
        self.folder = os.path.join(root, name + "_dynamic_explanation")
        self.dataset = dataset
        self.filtered_data = Data(
            edge_index=dataset.target_edge_index, edge_type=dataset.target_edge_type
        )
        self.hops = hops
        self.split = split

        if self.split == "train":
            self.split_data = self.dataset[0]
        elif self.split == "valid":
            self.split_data = self.dataset[1]
        else:
            self.split_data = self.dataset[2]

        edge_index = torch.cat(
            (
                self.split_data.target_edge_index,
                self.split_data.target_edge_index.flip(0),
            ),
            dim=1,
        )
        edge_type = torch.cat(
            (
                self.split_data.target_edge_type,
                self.split_data.target_edge_type + dataset.num_relations // 2,
            ),
            dim=0,
        )
        self.triples = torch.cat([edge_index, edge_type.unsqueeze(0)]).t()

    def __len__(self):
        return self.triples.shape[0]

    def __getitem__(self, idx):
        """
        We need to get the following:
        - edge_index within the k-hop neighborhood of the center node (not relabeled)
        - edge_mask identifying which edges they are
        -
        """
        triplet = self.triples[idx]

        # mask the target edge
        if self.split == "train":
            masked_data = remove_target_edge(
                self.dataset.num_relations // 2,
                self.split_data,
                triplet[0].unsqueeze(0),
                triplet[1].unsqueeze(0),
                triplet[2].unsqueeze(0),
            )
        else:
            masked_data = self.split_data

        # if the idx is the first half of the triples: tail-batch
        if idx < self.split_data.target_edge_index.shape[1]:
            mode = "tail-batch"
            positive_arg, center_node, eval_rel = triplet

            arg = 0  # 0 means we will later look at edges that has the node in the source when creating filter
            candidate_arg = (
                1  # 1 means we will later look at the candidates in the target
            )

        # if the idx is the latter half of the triples: head batch
        else:
            mode = "head-batch"
            center_node, positive_arg, eval_rel = triplet

            arg = 1  # 1 means we will later look at edges that has the node in the target when creating filter
            candidate_arg = (
                0  # 0 means we will later look at the candidates in the source
            )

        center_node = center_node.unsqueeze(0)
        if center_node in masked_data.edge_index:
            selected_nodes, edge_index, _, edge_mask = k_hop_subgraph(
                center_node, self.hops, masked_data.edge_index, relabel_nodes=False
            )
        else:
            selected_nodes = center_node
            edge_index = torch.tensor([[], []], dtype=torch.long)
            edge_mask = torch.zeros(masked_data.edge_type.shape, dtype=torch.bool)
