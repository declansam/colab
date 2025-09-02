import os
import pdb
import torch
from itertools import chain
import copy
from hetero_rgcn.data_processing import *
from torch_geometric.data import Data, InMemoryDataset

class HeteroDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name=name
        self.dataset_dir = 'hetero_datasets'
        self.root_dir = root
        dir = os.path.join(root, self.name+"_dataset")
        if not os.path.isdir(dir):
            os.makedirs(dir)
        super().__init__(dir, transform, pre_transform)
        data, groundtruth_data, dicts = torch.load(self.processed_paths[0])
        self.data, self.slices = data
        self.groundtruth_data = groundtruth_data
        self.ntype2id, self.rel2id = dicts
        self.num_relations = len(self.rel2id)

    @property
    def raw_file_names(self):
        return ''
    
    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        '''
        The data is in a DGL heterogenous datatype.
        We will make this into a PyG homogenous datatype, where the edge type and node type is stored separately.
        '''
        g, processed_g, pred_pair_to_edge_labels, pred_pair_to_path_labels = load_dataset(self.root_dir, self.name, val_ratio=0.1, test_ratio=0.2)
        # the message-passing data
        mp_g = processed_g[0]
        # list to hold the converted PyG data
        pyg_data = []
        # stores the number of nodes per ntype
        num_nodes_per_ntype = []
        # gives the offset for each ntype
        offset = {}
        ntype2id={}
        rel2id={}
        # node features
        ntypes = []
        for i, ntype in enumerate(mp_g.ntypes):
            # the cumulative sum upto this node
            prev_cum_sum = sum(num_nodes_per_ntype)
            num_nodes_per_ntype.append(mp_g.num_nodes(ntype))
            offset[ntype] = prev_cum_sum
            ntype2id[ntype] = i
            ntypes.extend([i]*mp_g.num_nodes(ntype))

                


        for g in processed_g: # for every graph
            combined_edge_index = torch.tensor([[], []], dtype=torch.int64)
            combined_edge_type = torch.tensor([], dtype=torch.int64)
            
            for canonical_etype in g.canonical_etypes:
                srctype, etype, dsttype = canonical_etype
                if etype not in rel2id.keys():
                    rel2id[etype] = len(rel2id)
                src, tgt = g.edges(etype=canonical_etype)
                edge_index = torch.stack((src, tgt))
                edge_index[0]+=offset[srctype]
                edge_index[1]+=offset[dsttype]
                edge_type = torch.tensor([rel2id[etype]]).repeat(edge_index.size(1))
                combined_edge_index = torch.cat((combined_edge_index, edge_index), dim=1)
                combined_edge_type = torch.cat((combined_edge_type, edge_type))

            
            pyg_data.append(Data(x=torch.arange(g.num_nodes()),
                                 ntypes=torch.tensor(ntypes), 
                                 edge_index=combined_edge_index, 
                                 edge_type=combined_edge_type, 
                                 num_nodes=g.num_nodes()))
            
        # mp_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g
        train_data = copy.copy(pyg_data[0])
        train_data.target_edge_index = pyg_data[1].edge_index
        train_data.target_edge_type = pyg_data[1].edge_type

        train_data.neg_target_edge_index = pyg_data[2].edge_index
        train_data.neg_target_edge_type = pyg_data[2].edge_type
        train_data.split = 'train'

        valid_data = copy.copy(pyg_data[0])
        valid_data.target_edge_index = pyg_data[3].edge_index
        valid_data.target_edge_type = pyg_data[3].edge_type
        valid_data.neg_target_edge_index = pyg_data[4].edge_index
        valid_data.neg_target_edge_type = pyg_data[4].edge_type
        valid_data.split = 'valid'

        test_data = copy.copy(pyg_data[0])
        test_data.target_edge_index = pyg_data[5].edge_index
        test_data.target_edge_type = pyg_data[5].edge_type
        test_data.neg_target_edge_index = pyg_data[6].edge_index
        test_data.neg_target_edge_type = pyg_data[6].edge_type
        test_data.split = 'test'

        # translate the ground truths to pyg format
        groundtruth_edge_index = []
        groundtruth_edge_type = []
        groundtruth_id = []
        posEdge2id = {}
        posEdge2Path = {}
        for i, src_tgt in enumerate(pred_pair_to_edge_labels.keys()):
            edge_labels = pred_pair_to_edge_labels[src_tgt]
            path_labels = pred_pair_to_path_labels[src_tgt]
            combined_edge_index = torch.tensor([[], []], dtype=torch.int64)
            combined_edge_type = torch.tensor([], dtype=torch.int64)
            for canonical_etype, edges in edge_labels.items():
                srctype, etype, dsttype = canonical_etype
                edge_index = torch.stack(edges)
                edge_index[0]+=offset[srctype]
                edge_index[1]+=offset[dsttype]
                edge_type = torch.tensor([rel2id[etype]]).repeat(edge_index.size(1))
                combined_edge_index = torch.cat((combined_edge_index, edge_index), dim=1)
                combined_edge_type = torch.cat((combined_edge_type, edge_type))
            groundtruth_edge_index.append(combined_edge_index)
            groundtruth_edge_type.append(combined_edge_type)
            groundtruth_id.append(torch.tensor([i]).repeat(combined_edge_type.size(0)))

            paths = []
            for path in path_labels:
                path_edge_index = torch.tensor([[], []], dtype=torch.int64)
                path_edge_type = torch.tensor([], dtype=torch.int64)
                for edge in path:
                    canonical_etype, srcid, dstid = edge
                    srctype, etype, dsttype = canonical_etype
                    srcid+=offset[srctype]
                    dstid+=offset[dsttype]
                    edge = torch.tensor([srcid, dstid]).unsqueeze(1)
                    edge_type = torch.tensor([rel2id[etype]])
                    path_edge_index = torch.cat((path_edge_index, edge), dim=1)
                    path_edge_type = torch.cat((path_edge_type, edge_type))
                paths.append((path_edge_index, path_edge_type))

            srctype, srcid = src_tgt[0]
            srcid+=offset[srctype]
            dsttype, dstid = src_tgt[1]
            dstid+=offset[dsttype]
            posEdge = (srcid, dstid)

            posEdge2id[posEdge] = i
            posEdge2Path[posEdge] = paths
        
        groundtruth_edge_index = torch.cat(groundtruth_edge_index, dim=1)
        groundtruth_edge_type = torch.cat(groundtruth_edge_type)
        groundtruth_id = torch.cat(groundtruth_id)
        groundtruth_data = Data(gt_edge_index = groundtruth_edge_index,
                                gt_edge_type = groundtruth_edge_type,
                                gt_id = groundtruth_id,
                                posEdge2id = posEdge2id,
                                posEdge2Path = posEdge2Path)

        torch.save((self.collate([train_data, valid_data, test_data]),
                    groundtruth_data,
                    (ntype2id, rel2id)), self.processed_paths[0])
