import os
import sys
import pprint
from itertools import chain
import torch
from torch_geometric.data import Data
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import tasks, util


def load_id2name(dataset, cls):
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

    id2entity = {}
    id2relation = {}
    for k, v in entity2id.items():
        id2entity[v] = k
    for k, v in relation2id.items():
        id2relation[v] = k

    return id2entity, id2relation


def get_topk_path(model, 
                  test_data, 
                  triplet, 
                  id2entity, 
                  id2relation, 
                  filtered_data=None, 
                  max_per_rel_counts=None, 
                  max_per_rel=None):
    topk_paths = {}
    disconnection_count = 0
    path_count = 0
    valid_count = 0

    num_relation = len(id2relation)
    triplet = triplet.unsqueeze(0)
    inverse = triplet[:, [1, 0, 2]]
    inverse[:, 2] += num_relation

    samples = (triplet, inverse)
    
    counts = 0
    for sample in samples:
        h, t, r = sample.squeeze(0).tolist()
        h_name = id2entity[h]
        t_name = id2entity[t]
        r_name = id2relation[r % num_relation]
        if r >= num_relation:
            r_name += "_inv"
        if max_per_rel is not None:
            if r_name not in max_per_rel_counts.keys():
                max_per_rel_counts[r_name] = 0
            if max_per_rel_counts[r_name] >= max_per_rel: # we have reached the max. triple per rel
                counts+=1
    
    if counts == 2: # if we are skipping for both
        return topk_paths, disconnection_count, path_count, valid_count, max_per_rel_counts

    model.eval()
    t_batch, h_batch = tasks.all_negative(test_data, triplet)
    t_pred = model(test_data, t_batch, triplet)
    h_pred = model(test_data, h_batch, triplet)

    if filtered_data is None:
        t_mask, h_mask = tasks.strict_negative_mask(test_data, triplet)
    else:
        t_mask, h_mask = tasks.strict_negative_mask(filtered_data, triplet)
    pos_h_index, pos_t_index, pos_r_index = triplet.unbind(-1)
    t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask).squeeze(0)
    h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask).squeeze(0)

    logger.warning("")
    rankings = (t_ranking, h_ranking)

    for sample, ranking in zip(samples, rankings):
        if ranking > 100: # if the rank is greater than 100, the path is likely to be useless. Move on.
            continue

        h, t, r = sample.squeeze(0).tolist()
        h_name = id2entity[h]
        t_name = id2entity[t]
        r_name = id2relation[r % num_relation]
        if r >= num_relation:
            r_name += "_inv"
        logger.warning(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        logger.warning("rank(%s | %s, %s) = %g" % (t_name, h_name, r_name, ranking))

        if max_per_rel is not None:
            if r_name not in max_per_rel_counts.keys():
                max_per_rel_counts[r_name] = 0
            if max_per_rel_counts[r_name] >= max_per_rel: # we have reached the max. triple per rel
                continue
            max_per_rel_counts[r_name] += 1 # increment the count

        topk_paths[(h_name, r_name, t_name)] = []

        paths, weights = model.visualize(test_data, sample)
        for path, weight in zip(paths, weights):
            path_count += 1
            prev_tail = None
            disconnect = False # whether the path is disconnected
            triplets = []
            relations = [] # keep the list of relations in the paths
            for h, t, r in path:
                h_name_ = id2entity[h]
                t_name_ = id2entity[t]
                r_name_ = id2relation[r % num_relation]
                if r >= num_relation:
                    r_name_ += "_inv"
                triplets.append("<%s, %s, %s>" % (h_name_, r_name_, t_name_))
                relations.append(r_name_)
                
                if (prev_tail is not None) and (h_name_ != prev_tail): # checking if there is any disconnection
                    triplets.append("* Disconnected *")
                    disconnect = True
                    disconnection_count += 1
                    break
                prev_tail = t_name_ # assign the tail as the prev tail
            if not disconnect: # only add paths that are connected
                topk_paths[(h_name, r_name, t_name)].append((relations, weight, ranking.item()))
                valid_count += 1
            logger.warning("weight: %g\n\t%s" % (weight, " ->\n\t".join(triplets)))
    
    return topk_paths, disconnection_count, path_count, valid_count, max_per_rel_counts


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    logger.warning("Config file: %s" % args.config)
    logger.warning(pprint.pformat(cfg))

    cls = cfg.dataset["class"]
    dataset = util.build_dataset(cfg)
    id2entity, id2relation = load_id2name(dataset, cls)
    cfg.model.num_relation = dataset.num_relations
    model = util.build_model(cfg)

    device = util.get_device(cfg)
    model = model.to(device)
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)
    # use the whole dataset for filtered ranking
    filtered_data = Data(edge_index=dataset._data.target_edge_index, edge_type=dataset._data.target_edge_type, num_nodes=train_data.num_nodes)
    filtered_data = filtered_data.to(device)

    path_count = 0
    disconnection_count = 0
    valid_count = 0
    if "max_per_rel" in cfg:  # max number of triples per rel
        max_per_rel_counts = {}
        max_per_rel = cfg.max_per_rel
    else:
        max_per_rel_counts, max_per_rel = None, None

    # get the topk paths for each split
    for data, split in zip([train_data], ["train"]):
        test_triplets = torch.cat([data.target_edge_index, data.target_edge_type.unsqueeze(0)]).t()

        topk_paths = {} 
        # a dictionary of key: (h, r, t) and value lists of tuples.
        # tuple[0] will be the path and tuple[1] will be its value and tuple[2] will be the predicted rank for that triple.
        for i in tqdm(range(test_triplets.shape[0]), desc=f"Obtaining TopK paths for {split}"): # for every triplet
            paths, d_count, p_count, v_count, max_per_rel_counts = get_topk_path(model, 
                                                                                data, 
                                                                                test_triplets[i], 
                                                                                id2entity, 
                                                                                id2relation, 
                                                                                filtered_data=filtered_data, 
                                                                                max_per_rel_counts=max_per_rel_counts,
                                                                                max_per_rel=max_per_rel)
            path_count += p_count
            disconnection_count += d_count
            valid_count += v_count
            topk_paths.update(paths)
            # logger.warning(f'Total Paths: {path_count}, Disconnected Paths: {disconnection_count}, Valid Paths: {valid_count}')

        logger.warning(f'* Total Paths: {path_count}, Disconnected Paths: {disconnection_count}, Valid Paths: {valid_count} *')
        torch.save(topk_paths, os.path.join(working_dir, f"{split}_topk_paths.pt"))
