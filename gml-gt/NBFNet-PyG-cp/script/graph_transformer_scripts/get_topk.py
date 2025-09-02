import os
import sys
import math
import pprint
import pdb
from tqdm import tqdm
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

vocab_file = os.path.join(os.path.dirname(__file__), "../data/fb15k237_entity.txt")
vocab_file = os.path.abspath(vocab_file)

@torch.no_grad()
def test_and_get_topk(cfg, model, test_data, id2entity, id2relation, topk, working_dir, split, filtered_data=None, get_filtered_topk = False):
    world_size = util.get_world_size()
    rank = util.get_rank()

    test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

    model.eval()
    rankings = []
    num_negatives = []
    topk_entities = {} # a dict of key (head, rel) and list of topk ranked entities. we don't do filtering.
    for batch in tqdm(test_loader):
        t_batch, h_batch = tasks.all_negative(test_data, batch)
        # do predictions for tail-batch mode only
        t_pred = model(test_data, t_batch)
        h_pred = model(test_data, h_batch)

        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)

        if get_filtered_topk:
            negatives = t_mask.clone()
            negatives[torch.arange(pos_t_index.shape[0]), pos_t_index] = True
            topk_t = tasks.get_topk(t_pred, topk, negatives==False)
        else:
            topk_t = tasks.get_topk(t_pred, topk)

        for i in range(len(batch)):
            head, tail, rel = batch[i]
            head = head.item()
            rel = rel.item()
            tail = tail.item()
            if get_filtered_topk:
                topk_entities[(id2entity[head], id2relation[rel], id2entity[tail])] = [id2entity[e.item()] for e in topk_t[i]]
            else:
                topk_entities[(id2entity[head], id2relation[rel])] = [id2entity[e.item()] for e in topk_t[i]]

        if get_filtered_topk:
            negatives = h_mask.clone()
            negatives[torch.arange(pos_h_index.shape[0]), pos_h_index] = True
            topk_h = tasks.get_topk(h_pred, topk, negatives==False)
        else:
            topk_h = tasks.get_topk(h_pred, topk)

        for i in range(len(batch)):
            head, tail, rel = batch[i]
            tail = tail.item()
            rel = rel.item()
            head = head.item()
            if get_filtered_topk:
                topk_entities[(id2entity[tail], id2relation[rel]+"_inv", id2entity[head])] = [id2entity[e.item()] for e in topk_h[i]]
            else:
                topk_entities[(id2entity[tail], id2relation[rel]+"_inv")] = [id2entity[e.item()] for e in topk_h[i]]

        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)

        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

    ranking = torch.cat(rankings)
    num_negative = torch.cat(num_negatives)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative
    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)

    # save the topk entities
    if get_filtered_topk:
        file_name = f"{split}_top{topk}_filtered.pt"
    else:
        file_name = f"{split}_top{topk}.pt"
    torch.save(topk_entities, os.path.join(working_dir, file_name))

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
    mrr = (1 / all_ranking.float()).mean()

    return mrr


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)

    assert util.get_world_size() == 1, "TopK inference only implemented for 1 GPU max."
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
    id2entity, id2relation = util.load_id2name(dataset, cls)

    cfg.model.num_relation = dataset.num_relations
    model = util.build_model(cfg)
    model.eval()

    device = util.get_device(cfg)
    model = model.to(device)
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)
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

    if "get_filtered_topk" in cfg:
        get_filtered_topk = cfg.get_filtered_topk
    else:
        get_filtered_topk = False
    # if util.get_rank() == 0:
    #     logger.warning(separator)
    #     logger.warning("Evaluate on train")
    # test_and_get_topk(cfg, model, train_data, id2entity, id2relation, topk, working_dir, "train", filtered_data=filtered_data)
    # if util.get_rank() == 0:
    #     logger.warning(separator)
    #     logger.warning("Evaluate on valid")
    # test_and_get_topk(cfg, model, valid_data, id2entity, id2relation, topk, working_dir, "valid", filtered_data=filtered_data)
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")
    test_and_get_topk(cfg, model, test_data, id2entity, id2relation, topk, working_dir, "test", filtered_data=filtered_data, get_filtered_topk=get_filtered_topk)