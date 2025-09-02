import os
import pdb
import torch
from itertools import chain
from torch_geometric.data import Data, InMemoryDataset, download_url


class IndRelLinkPredDataset(InMemoryDataset):

    urls = {
        "FB15k-237": [
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/test.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/valid.txt",
        ],
        "WN18RR": [
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/test.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/valid.txt",
        ],
    }

    def __init__(self, root, name, version, transform=None, pre_transform=None):
        self.name = name
        self.version = version
        assert name in ["FB15k-237", "WN18RR"]
        assert version in ["v1", "v2", "v3", "v4"]
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, self.version, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, self.version, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"

    @property
    def raw_file_names(self):
        return ["train_ind.txt", "test_ind.txt", "train.txt", "valid.txt"]

    def download(self):
        for url, path in zip(self.urls[self.name], self.raw_paths):
            download_path = download_url(url % self.version, self.raw_dir)
            os.rename(download_path, path)

    def process(self):
        test_files = self.raw_paths[:2]
        train_files = self.raw_paths[2:]

        inv_train_entity_vocab = {}
        inv_test_entity_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []

        for txt_file in train_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[h_token] = len(inv_train_entity_vocab)
                    h = inv_train_entity_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[t_token] = len(inv_train_entity_vocab)
                    t = inv_train_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        for txt_file in test_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[h_token] = len(inv_test_entity_vocab)
                    h = inv_test_entity_vocab[h_token]
                    assert r_token in inv_relation_vocab
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[t_token] = len(inv_test_entity_vocab)
                    t = inv_test_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)
        triplets = torch.tensor(triplets)

        edge_index = triplets[:, :2].t()
        edge_type = triplets[:, 2]
        num_relations = int(edge_type.max()) + 1

        train_fact_slice = slice(None, sum(num_samples[:1]))
        test_fact_slice = slice(sum(num_samples[:2]), sum(num_samples[:3]))
        train_fact_index = edge_index[:, train_fact_slice]
        train_fact_type = edge_type[train_fact_slice]
        test_fact_index = edge_index[:, test_fact_slice]
        test_fact_type = edge_type[test_fact_slice]
        # add flipped triplets for the fact graphs
        train_fact_index = torch.cat(
            [train_fact_index, train_fact_index.flip(0)], dim=-1
        )
        train_fact_type = torch.cat([train_fact_type, train_fact_type + num_relations])
        test_fact_index = torch.cat([test_fact_index, test_fact_index.flip(0)], dim=-1)
        test_fact_type = torch.cat([test_fact_type, test_fact_type + num_relations])

        train_slice = slice(None, sum(num_samples[:1]))
        valid_slice = slice(sum(num_samples[:1]), sum(num_samples[:2]))
        test_slice = slice(sum(num_samples[:3]), sum(num_samples))
        train_data = Data(
            edge_index=train_fact_index,
            edge_type=train_fact_type,
            num_nodes=len(inv_train_entity_vocab),
            target_edge_index=edge_index[:, train_slice],
            target_edge_type=edge_type[train_slice],
        )
        valid_data = Data(
            edge_index=train_fact_index,
            edge_type=train_fact_type,
            num_nodes=len(inv_train_entity_vocab),
            target_edge_index=edge_index[:, valid_slice],
            target_edge_type=edge_type[valid_slice],
        )
        test_data = Data(
            edge_index=test_fact_index,
            edge_type=test_fact_type,
            num_nodes=len(inv_test_entity_vocab),
            target_edge_index=edge_index[:, test_slice],
            target_edge_type=edge_type[test_slice],
        )

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save(
            (self.collate([train_data, valid_data, test_data])), self.processed_paths[0]
        )

    def __repr__(self):
        return "%s()" % self.name


class WordNet18RR(InMemoryDataset):
    r"""The WordNet18RR dataset from the `"Convolutional 2D Knowledge Graph
    Embeddings" <https://arxiv.org/abs/1707.01476>`_ paper, containing 40,943
    entities, 11 relations and 93,003 fact triplets.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = (
        "https://raw.githubusercontent.com/villmow/"
        "datasets_knowledge_embedding/master/WN18RR/text"
    )

    edge2id = {
        "_also_see": 0,
        "_derivationally_related_form": 1,
        "_has_part": 2,
        "_hypernym": 3,
        "_instance_hypernym": 4,
        "_member_meronym": 5,
        "_member_of_domain_region": 6,
        "_member_of_domain_usage": 7,
        "_similar_to": 8,
        "_synset_domain_topic_of": 9,
        "_verb_group": 10,
    }

    def __init__(self, root, transform=None, pre_transform=None):
        super(WordNet18RR, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["train.txt", "valid.txt", "test.txt"]

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        for filename in self.raw_file_names:
            download_url(f"{self.url}/{filename}", self.raw_dir)

    def process(self):
        node2id, idx = {}, 0

        srcs, dsts, edge_types = [], [], []
        for path in self.raw_paths:
            with open(path, "r") as f:
                data = f.read().split()

                src = data[::3]
                dst = data[2::3]
                edge_type = data[1::3]

                for i in chain(src, dst):
                    if i not in node2id:
                        node2id[i] = idx
                        idx += 1

                src = [node2id[i] for i in src]
                dst = [node2id[i] for i in dst]
                edge_type = [self.edge2id[i] for i in edge_type]

                srcs.append(torch.tensor(src, dtype=torch.long))
                dsts.append(torch.tensor(dst, dtype=torch.long))
                edge_types.append(torch.tensor(edge_type, dtype=torch.long))

        src = torch.cat(srcs, dim=0)
        dst = torch.cat(dsts, dim=0)
        edge_type = torch.cat(edge_types, dim=0)

        train_mask = torch.zeros(src.size(0), dtype=torch.bool)
        train_mask[: srcs[0].size(0)] = True
        val_mask = torch.zeros(src.size(0), dtype=torch.bool)
        val_mask[srcs[0].size(0) : srcs[0].size(0) + srcs[1].size(0)] = True
        test_mask = torch.zeros(src.size(0), dtype=torch.bool)
        test_mask[srcs[0].size(0) + srcs[1].size(0) :] = True

        num_nodes = max(int(src.max()), int(dst.max())) + 1
        perm = (num_nodes * src + dst).argsort()

        edge_index = torch.stack([src[perm], dst[perm]], dim=0)
        edge_type = edge_type[perm]
        train_mask = train_mask[perm]
        val_mask = val_mask[perm]
        test_mask = test_mask[perm]

        data = Data(
            edge_index=edge_index,
            edge_type=edge_type,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_nodes=num_nodes,
        )

        if self.pre_transform is not None:
            data = self.pre_filter(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class TransductiveDataset(InMemoryDataset):

    delimiter = None

    def __init__(self, root, transform=None, pre_transform=None, **kwargs):
        super().__init__(root, transform, pre_transform)
        data, self.entity2id, self.rel2id = torch.load(self.processed_paths[0])
        self.data, self.slices = data

    @property
    def raw_file_names(self):
        return ["train.txt", "valid.txt", "test.txt"]

    def download(self):
        for url, path in zip(self.urls, self.raw_paths):
            download_path = download_url(url, self.raw_dir)
            os.rename(download_path, path)

    def load_file(self, triplet_file, inv_entity_vocab={}, inv_rel_vocab={}):

        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        with open(triplet_file, "r", encoding="utf-8") as fin:
            for l in fin:
                u, r, v = (
                    l.split()
                    if self.delimiter is None
                    else l.strip().split(self.delimiter)
                )
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

                triplets.append((u, v, r))

        return {
            "triplets": triplets,
            "num_node": len(inv_entity_vocab),  # entity_cnt,
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab,
        }

    # default loading procedure: process train/valid/test files, create graphs from them
    def process(self):

        train_files = self.raw_paths[:3]

        train_results = self.load_file(
            train_files[0], inv_entity_vocab={}, inv_rel_vocab={}
        )
        valid_results = self.load_file(
            train_files[1],
            train_results["inv_entity_vocab"],
            train_results["inv_rel_vocab"],
        )
        test_results = self.load_file(
            train_files[2],
            train_results["inv_entity_vocab"],
            train_results["inv_rel_vocab"],
        )

        # in some datasets, there are several new nodes in the test set, eg 123,143 YAGO train adn 123,182 in YAGO test
        # for consistency with other experimental results, we'll include those in the full vocab and num nodes
        num_node = test_results["num_node"]
        # the same for rels: in most cases train == test for transductive
        # for AristoV4 train rels 1593, test 1604
        num_relations = test_results["num_relation"]

        train_triplets = train_results["triplets"]
        valid_triplets = valid_results["triplets"]
        test_triplets = test_results["triplets"]

        train_target_edges = torch.tensor(
            [[t[0], t[1]] for t in train_triplets], dtype=torch.long
        ).t()
        train_target_etypes = torch.tensor([t[2] for t in train_triplets])

        valid_edges = torch.tensor(
            [[t[0], t[1]] for t in valid_triplets], dtype=torch.long
        ).t()
        valid_etypes = torch.tensor([t[2] for t in valid_triplets])

        test_edges = torch.tensor(
            [[t[0], t[1]] for t in test_triplets], dtype=torch.long
        ).t()
        test_etypes = torch.tensor([t[2] for t in test_triplets])

        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_etypes = torch.cat(
            [train_target_etypes, train_target_etypes + num_relations]
        )

        train_data = Data(
            edge_index=train_edges,
            edge_type=train_etypes,
            num_nodes=num_node,
            target_edge_index=train_target_edges,
            target_edge_type=train_target_etypes,
            num_relations=num_relations * 2,
            split="train",
        )
        valid_data = Data(
            edge_index=train_edges,
            edge_type=train_etypes,
            num_nodes=num_node,
            target_edge_index=valid_edges,
            target_edge_type=valid_etypes,
            num_relations=num_relations * 2,
            split="valid",
        )
        test_data = Data(
            edge_index=train_edges,
            edge_type=train_etypes,
            num_nodes=num_node,
            target_edge_index=test_edges,
            target_edge_type=test_etypes,
            num_relations=num_relations * 2,
            split="test",
        )

        # build graphs of relations
        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save(
            (
                self.collate([train_data, valid_data, test_data]),
                train_results["inv_entity_vocab"],
                train_results["inv_rel_vocab"],
            ),
            self.processed_paths[0],
        )

    def __repr__(self):
        return "%s()" % (self.name)

    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"


class NELL995(TransductiveDataset):

    # from the RED-GNN paper https://github.com/LARS-research/RED-GNN/tree/main/transductive/data/nell
    # the OG dumps were found to have test set leakages
    # Ultra makes the training set to be the facts+train files, but this could explode the training set.
    # therefore, we make the graph facts+train but train only using the triples in train.

    urls = [
        "https://raw.githubusercontent.com/LARS-research/RED-GNN/main/transductive/data/nell/facts.txt",
        "https://raw.githubusercontent.com/LARS-research/RED-GNN/main/transductive/data/nell/train.txt",
        "https://raw.githubusercontent.com/LARS-research/RED-GNN/main/transductive/data/nell/valid.txt",
        "https://raw.githubusercontent.com/LARS-research/RED-GNN/main/transductive/data/nell/test.txt",
    ]
    name = "nell995"
    num_relations = None

    @property
    def raw_file_names(self):
        return ["facts.txt", "train.txt", "valid.txt", "test.txt"]

    def download(self):
        super().download()  # on PyG 2.0.1, you have to explicitly write this otherwise it won't download

    def process(self):
        train_files = self.raw_paths[:4]

        facts_results = self.load_file(
            train_files[0], inv_entity_vocab={}, inv_rel_vocab={}
        )
        train_results = self.load_file(
            train_files[1],
            facts_results["inv_entity_vocab"],
            facts_results["inv_rel_vocab"],
        )
        valid_results = self.load_file(
            train_files[2],
            train_results["inv_entity_vocab"],
            train_results["inv_rel_vocab"],
        )
        test_results = self.load_file(
            train_files[3],
            train_results["inv_entity_vocab"],
            train_results["inv_rel_vocab"],
        )

        num_node = valid_results["num_node"]
        num_relations = train_results["num_relation"]

        fact_triplets = facts_results["triplets"] + train_results["triplets"]
        train_triplets = train_results["triplets"]
        valid_triplets = valid_results["triplets"]
        test_triplets = test_results["triplets"]

        fact_target_edges = torch.tensor(
            [[t[0], t[1]] for t in fact_triplets], dtype=torch.long
        ).t()
        fact_target_etypes = torch.tensor([t[2] for t in fact_triplets])

        train_target_edges = torch.tensor(
            [[t[0], t[1]] for t in train_triplets], dtype=torch.long
        ).t()
        train_target_etypes = torch.tensor([t[2] for t in train_triplets])

        valid_edges = torch.tensor(
            [[t[0], t[1]] for t in valid_triplets], dtype=torch.long
        ).t()
        valid_etypes = torch.tensor([t[2] for t in valid_triplets])

        test_edges = torch.tensor(
            [[t[0], t[1]] for t in test_triplets], dtype=torch.long
        ).t()
        test_etypes = torch.tensor([t[2] for t in test_triplets])

        train_edges = torch.cat([fact_target_edges, fact_target_edges.flip(0)], dim=1)
        train_etypes = torch.cat(
            [fact_target_etypes, fact_target_etypes + num_relations]
        )

        train_data = Data(
            edge_index=train_edges,
            edge_type=train_etypes,
            num_nodes=num_node,
            target_edge_index=train_target_edges,
            target_edge_type=train_target_etypes,
            num_relations=num_relations * 2,
            split="train",
        )
        valid_data = Data(
            edge_index=train_edges,
            edge_type=train_etypes,
            num_nodes=num_node,
            target_edge_index=valid_edges,
            target_edge_type=valid_etypes,
            num_relations=num_relations * 2,
            split="valid",
        )
        test_data = Data(
            edge_index=train_edges,
            edge_type=train_etypes,
            num_nodes=num_node,
            target_edge_index=test_edges,
            target_edge_type=test_etypes,
            num_relations=num_relations * 2,
            split="test",
        )

        # build graphs of relations
        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save(
            (
                self.collate([train_data, valid_data, test_data]),
                train_results["inv_entity_vocab"],
                train_results["inv_rel_vocab"],
            ),
            self.processed_paths[0],
        )


class YAGO310(TransductiveDataset):

    urls = [
        "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/train.txt",
        "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/valid.txt",
        "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/test.txt",
    ]
    name = "yago310"
    num_relations = None

    def download(self):
        super().download()  # on PyG 2.0.1, you have to explicitly write this otherwise it won't download

    def process(self):
        super().process()  # on PyG 2.0.1, you have to explicitly write this otherwise it won't download
