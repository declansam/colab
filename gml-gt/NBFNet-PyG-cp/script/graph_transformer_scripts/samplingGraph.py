import os
import sys
import math
import pprint
from tqdm import tqdm
from itertools import chain
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import tasks, util


separator = ">" * 30
line = "-" * 30


def get_dict(dataset, cls):
    if cls == "FB15k-237":
        with open(os.path.join(dataset.raw_dir, 'entities.dict')) as f:
            lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
            entity2id = {key: int(value) for value, key in lines}

        with open(os.path.join(dataset.raw_dir, 'relations.dict')) as f:
            lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
            relation2id = {key: int(value) for value, key in lines}

    elif cls == "WN18RR":
        entity2id, idx = {}, 0
        for path in dataset.raw_paths:
            with open(path) as f:
                edges = f.read().split()

                _src = edges[::3]
                _dst = edges[2::3]

                for i in chain(_src, _dst):
                    if i not in entity2id:
                        entity2id[i] = idx
                        idx += 1

        relation2id = dataset.edge2id

    elif cls in ["NELL995", "YAGO310"]:
        entity2id = dataset.entity2id
        relation2id = dataset.rel2id
    else:
        raise NotImplementedError

    return entity2id, relation2id

def create_graph(path_file, original_ent2id, rel2id):
    triples, target_edge = torch.load(path_file)

    ent2id = {}

    edge_index = []
    original_edge_index = []
    edge_type = []
    for triple in triples:
        for ent in [triple[0], triple[2]]:
            if ent not in ent2id.keys():
                ent2id[ent] = len(ent2id)
        edge_index.append([ent2id[triple[0]], ent2id[triple[2]]])
        original_edge_index.append([original_ent2id[triple[0]], original_ent2id[triple[2]]])
        edge_type.append(rel2id[triple[1]])

    edge_index = torch.tensor(edge_index).T
    original_edge_index = torch.tensor(original_edge_index).T
    edge_type = torch.tensor(edge_type)

    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1) # adding inverse edges
    edge_type = torch.cat([edge_type, edge_type + len(rel2id)]) # adding inverse edge attr
    original_edge_index = torch.cat([original_edge_index, original_edge_index.flip(0)], dim=-1)

    for ent in [target_edge[0], target_edge[2]]:
        if ent not in ent2id.keys():
            ent2id[ent] = len(ent2id)
    target_edge_index = [[ent2id[target_edge[0]], ent2id[target_edge[2]]]]
    original_target_edge_index = [[original_ent2id[target_edge[0]], original_ent2id[target_edge[2]]]]
    target_edge_type = [rel2id[target_edge[1]]]

    target_edge_index = torch.tensor(target_edge_index).T
    original_target_edge_index = torch.tensor(original_target_edge_index).T
    target_edge_type = torch.tensor(target_edge_type)

    sampled_data = Data(edge_index=edge_index, 
                        edge_type=edge_type, 
                        target_edge_index=target_edge_index, 
                        target_edge_type=target_edge_type, 
                        split='test',
                        num_nodes=len(ent2id))

    sampled_data_original_index = Data(edge_index=original_edge_index, 
                                        edge_type=edge_type, 
                                        target_edge_index=original_target_edge_index, 
                                        target_edge_type=target_edge_type, 
                                        split='test')
    
    original_nodes = []
    # get the original id of the entities included, in the order they appear in the dict.
    for ent, _ in ent2id.items():
        original_nodes.append(original_ent2id[ent])

    return sampled_data, sampled_data_original_index, ent2id, original_nodes

@torch.no_grad()
def test_sample(cfg, model, test_data, test_data_original_index, original_nodes, filtered_data=None, working_dir=None, split='test', id2entity = None, id2relation = None, topk='all'):
    world_size = util.get_world_size()
    rank = util.get_rank()

    test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)
    
    test_original_triplets = torch.cat([test_data_original_index.target_edge_index, test_data_original_index.target_edge_type.unsqueeze(0)]).t()
    sampler_original = torch_data.DistributedSampler(test_original_triplets, world_size, rank)
    test_original_loader = torch_data.DataLoader(test_original_triplets, cfg.train.batch_size, sampler=sampler_original)

    if rank == 0:
        pbar = tqdm(total=len(test_loader))

    model.eval()
    rankings = []
    num_negatives = []
    modes = [] # logging the mode (0 for tail, 1 for head)
    rels = [] # logging the rel types.
    topk_entities = {} # a dict of key (head, rel) and list of topk ranked entities. we don't do filtering.
    for batch, original_batch in zip(test_loader, test_original_loader):
        t_batch, h_batch = tasks.all_negative(test_data, batch)
        t_pred = model(test_data, t_batch)
        h_pred = model(test_data, h_batch)

        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, original_batch)
            # filter only the list of nodes in test_data
            t_mask = t_mask[:, original_nodes]
            h_mask = h_mask[:, original_nodes]

        pos_h_index, pos_t_index, pos_r_index = batch.t()
        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)

        topk_t = tasks.get_topk(t_pred, topk)
        for i in range(len(batch)):
            head, tail, rel = batch[i]
            head = head.item()
            rel = rel.item()
            topk_entities[(id2entity[head], id2relation[rel])] = [id2entity[e.item()] for e in topk_t[i]]
            
        topk_h = tasks.get_topk(h_pred, topk)
        for i in range(len(batch)):
            head, tail, rel = batch[i]
            tail = tail.item()
            rel = rel.item()
            topk_entities[(id2entity[tail], id2relation[rel]+"_inv")] = [id2entity[e.item()] for e in topk_h[i]]

        if rank == 0:
            for i in range(len(batch)):
                head, tail, rel = batch[i]
                head, tail, rel = head.item(), tail.item(), rel.item()
                print(f"Query {id2entity[head], id2relation[rel]}: ")
                entities = topk_entities[(id2entity[head], id2relation[rel])]
                ent_ids = topk_t[i]
                for j, (ent_name, e_id) in enumerate(zip(entities, ent_ids)):
                    if (t_mask[i, e_id] == False) and not (e_id.item() == tail):
                        print(j+1, ent_name, 'Filtered')
                    elif e_id.item() == tail:
                        print(j+1, ent_name, 'Answer')
                    else:
                        print(j+1, ent_name)
                print(f"* The filtered tail-batch rank is {t_ranking[i]} *")
            
                print(f"Query {id2entity[tail], id2relation[rel]+'_inv'}: ")
                entities = topk_entities[(id2entity[tail], id2relation[rel]+"_inv")]
                ent_ids = topk_h[i]
                for j, (ent_name, e_id) in enumerate(zip(entities, ent_ids)):
                    if h_mask[i, e_id] == False and not e_id.item() == head:
                        print(j+1, ent_name, 'Filtered')
                    elif e_id.item() == head:
                        print(j+1, ent_name, 'Answer')
                    else:
                        print(j+1, ent_name)
                print(f"* The filtered head-batch rank is {h_ranking[i]} *")

        if rank == 0:   
            torch.save(topk_entities, os.path.join(working_dir, f"sample_top{topk}.pt"))
            print(f"Check {os.path.join(working_dir, f'sample_top{topk}.pt')} for in-depth topk")

        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]
        rels += [batch[:, -1], batch[:, -1]] # get the rels for both tail and head predictions
        modes += [torch.zeros(t_ranking.shape), torch.ones(h_ranking.shape)] # log whether it was for tail prediction or head prediction

        if rank == 0:
            pbar.update(1)
    
    if rank == 0:
        pbar.close()

    ranking = torch.cat(rankings)
    num_negative = torch.cat(num_negatives)
    rels = torch.cat(rels)
    modes = torch.cat(modes)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative
    all_rels = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_rels[cum_size[rank] - all_size[rank]: cum_size[rank]] = rels
    all_modes = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_modes[cum_size[rank] - all_size[rank]: cum_size[rank]] = modes

    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_rels, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_modes, op=dist.ReduceOp.SUM)

    if rank == 0:
        for metric in cfg.task.metric:
            if metric == "mr":
                score = all_ranking.float().mean()
            elif metric == "mrr":
                score = (1 / all_ranking.float()).mean()
            elif metric.startswith("hits@"):
                values = metric[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    # unbiased estimation
                    fp_rate = (all_ranking - 1).float() / all_num_negative
                    score = 0
                    for i in range(threshold):
                        # choose i false positive from num_sample - 1 negatives
                        num_comb = math.factorial(num_sample - 1) / \
                                   math.factorial(i) / math.factorial(num_sample - i - 1)
                        score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                    score = score.mean()
                else:
                    score = (all_ranking <= threshold).float().mean()
            logger.warning("%s: %g" % (metric, score))
        
        if working_dir is not None:
            data = {'Ranking': all_ranking.tolist(),
                    'Rel': all_rels.tolist(),
                    'Mode': all_modes.tolist()
                    }
            torch.save(data, os.path.join(working_dir, f'{split}_output.pt'))
            print(f"Check {os.path.join(working_dir, f'{split}_output.pt')} for in-depth output")

    mrr = (1 / all_ranking.float()).mean()

    return mrr


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    assert util.get_world_size() == 1, "This module is implemented for 1 GPU max."
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
    is_inductive = cfg.dataset["class"].startswith("Ind")

    cls = cfg.dataset["class"]
    dataset = util.build_dataset(cfg)
    ent2id, rel2id = get_dict(dataset, cls)

    cfg.model.num_relation = dataset.num_relations
    model = util.build_model(cfg)

    device = util.get_device(cfg)
    model = model.to(device)
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)
    
    if not args.file_name:
        raise Exception('Pass the file name for the sampled data using --file_name')
    
    sampled_data, sampled_data_original_index, ent2id, original_nodes = create_graph(args.file_name, ent2id, rel2id)
    sampled_data.to(device)
    sampled_data_original_index.to(device)

    if is_inductive:
        # for inductive setting, use only the test fact graph for filtered ranking
        filtered_data = None
    else:
        # for transductive setting, use the whole graph for filtered ranking
        filtered_data = Data(edge_index=dataset._data.target_edge_index, edge_type=dataset._data.target_edge_type, num_nodes=train_data.num_nodes)
        filtered_data = filtered_data.to(device)
    
    if "topk" in cfg: # how many entities to retrieve
        topk = cfg.topk
    else:
        topk = 100

    id2rel, id2ent = {}, {}
    for k, v in rel2id.items():
        id2rel[v] = k
    for k, v in ent2id.items():
        id2ent[v] = k

    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on sampled data")
    test_sample(cfg, model, sampled_data, sampled_data_original_index, original_nodes, filtered_data=filtered_data, working_dir=working_dir, split='test', id2entity=id2ent, id2relation=id2rel, topk=topk)