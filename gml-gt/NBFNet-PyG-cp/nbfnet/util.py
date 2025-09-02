import os
import sys
import ast
import time
import logging
import argparse
from itertools import chain
import yaml
import jinja2
from jinja2 import meta
import easydict
import copy

import torch
from torch import distributed as dist
from torch_geometric.data import Data
from torch_geometric.datasets import RelLinkPredDataset
from hetero_rgcn import hetero_datasets
from hetero_rgcn import rgcn

from nbfnet import models, datasets, models_expl


logger = logging.getLogger(__file__)


def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    tree = env.parse(raw)
    vars = meta.find_undeclared_variables(tree)
    return vars


def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return cfg


def literal_eval(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return string


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument(
        "-s", "--seed", help="random seed for PyTorch", type=int, default=1024
    )
    parser.add_argument(
        "--file_name", help="file name for a sampled graph", type=str, default=""
    )
    parser.add_argument(
        "--hyper_search",
        help="whether to do hyper search",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--num_runs", help="how many training of NBFNet to do", type=int, default=1
    )
    parser.add_argument("--use_cfg", action="store_true", default=False)
    parser.add_argument("--hyper_run_id", type=int, default=-1)
    parser.add_argument("--eval_on_edge_drop", action="store_true", default=False)
    parser.add_argument("--randomized_edge_drop", default=-1, type=float)
    parser.add_argument(
        "--train_config", help="yaml configuration file", type=str, default=None
    )

    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, required=True)
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: literal_eval(v) for k, v in vars._get_kwargs()}

    return args, vars


def get_root_logger(file=True):
    format = "%(asctime)-10s %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(format=format, datefmt=datefmt)
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    if file:
        handler = logging.FileHandler("log.txt")
        format = logging.Formatter(format, datefmt)
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def change_logger_file(logger, file=True):
    # call this function to change the logging file location to current_dir/log.txt
    format = "%(asctime)-10s %(message)s"
    datefmt = "%H:%M:%S"

    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)

    if file:
        handler = logging.FileHandler("log.txt")
        format = logging.Formatter(format, datefmt)
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0


def get_world_size(num_gpus=None):
    if dist.is_initialized():
        return dist.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    if num_gpus:
        return num_gpus
    return 1


def synchronize():
    if get_world_size() > 1:
        dist.barrier()


def get_device(cfg):
    if cfg.train.gpus:
        device = torch.device(cfg.train.gpus[get_rank()])
    else:
        device = torch.device("cpu")
    return device


def create_working_directory(cfg):
    file_name = "working_dir.tmp"
    world_size = get_world_size()
    if cfg.train.gpus is not None and len(cfg.train.gpus) != world_size:
        error_msg = "World size is %d but found %d GPUs in the argument"
        if world_size == 1:
            error_msg += ". Did you launch with `python -m torch.distributed.launch`?"
        raise ValueError(error_msg % (world_size, len(cfg.train.gpus)))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", init_method="env://")

    working_dir = os.path.join(
        os.path.expanduser(cfg.output_dir),
        cfg.model["class"],
        cfg.dataset["class"],
        time.strftime("%Y-%m-%d-%H-%M-%S"),
    )

    # synchronize working directory
    if get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(working_dir)
        os.makedirs(working_dir)
    synchronize()
    if get_rank() != 0:
        with open(file_name, "r") as fin:
            working_dir = fin.read()
    synchronize()
    if get_rank() == 0:
        os.remove(file_name)

    os.chdir(working_dir)
    return working_dir


def create_working_directory_per_run(run_id):
    file_name = "working_dir.tmp"
    working_dir = os.path.join(
        os.getcwd(),
        f"run_{run_id}",
    )
    # synchronize working directory
    if get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(working_dir)
        os.makedirs(working_dir)
    synchronize()
    if get_rank() != 0:
        with open(file_name, "r") as fin:
            working_dir = fin.read()
    synchronize()
    if get_rank() == 0:
        os.remove(file_name)

    os.chdir(working_dir)
    return working_dir


def remove_duplicate(all_query, all_items):
    unique_query, indices, counts = torch.unique(
        all_query, dim=0, return_inverse=True, return_counts=True
    )
    unique_items = torch.zeros(
        (unique_query.size(0), all_items.size(1)),
        dtype=all_items.dtype,
        device=all_items.device,
    )
    unique_items[indices, :] = all_items[torch.arange(all_items.size(0)), :]
    return unique_query, unique_items


def rowwise_isin(tensor_1, target_tensor):
    # make sure the rowwise_isin operation does not lead to OOM. Do batchwise if necesssary
    # `matches` will be of size (num_rows, num_edges, num_edges)
    # the max_budget is the max budget the cpu can allocate
    # therefore, calculate the max_rows we can process at the time
    max_budget = 10**11
    num_edges = tensor_1.size(1)
    max_rows = max_budget // (num_edges**2)

    batches = torch.split(tensor_1, max_rows)
    target_batches = torch.split(target_tensor, max_rows)

    results = []
    for i in range(len(batches)):
        t_1 = batches[i]
        target = target_batches[i]
        matches = t_1.unsqueeze(2) == target.unsqueeze(1)
        # result: boolean tensor of shape (N, K) where result[n, k] is torch.isin(tensor_1[n, k], target_tensor[n])
        result = torch.sum(matches, dim=2, dtype=torch.bool)
        results.append(result)
    result = torch.cat(results)
    return result


def build_dataset(cfg):
    dataset_cfg = copy.deepcopy(cfg)
    cls = dataset_cfg.dataset.pop("class")
    if hasattr(dataset_cfg.dataset, "combined_graph"):
        combined_graph = dataset_cfg.dataset.pop("combined_graph")
    else:
        combined_graph = False
    if cls == "FB15k-237":
        dataset = RelLinkPredDataset(name=cls, **dataset_cfg.dataset)
        data = dataset.data
        train_data = Data(
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            num_nodes=data.num_nodes,
            target_edge_index=data.train_edge_index,
            target_edge_type=data.train_edge_type,
            split="train",
        )
        valid_data = Data(
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            num_nodes=data.num_nodes,
            target_edge_index=data.valid_edge_index,
            target_edge_type=data.valid_edge_type,
            split="valid",
        )
        if (
            combined_graph
        ):  # a graph that combines the train and valid edges for inference on test set
            valid_edge_index = torch.cat(
                (data.valid_edge_index, data.valid_edge_index.flip(0)), dim=1
            )
            valid_edge_type = torch.cat(
                (
                    data.valid_edge_type,
                    data.valid_edge_type + dataset.num_relations // 2,
                ),
                dim=0,
            )
            test_data = Data(
                edge_index=torch.cat((data.edge_index, valid_edge_index), dim=1),
                edge_type=torch.cat((data.edge_type, valid_edge_type), dim=0),
                target_edge_index=data.test_edge_index,
                target_edge_type=data.test_edge_type,
                split="test",
            )
        else:
            test_data = Data(
                edge_index=data.edge_index,
                edge_type=data.edge_type,
                num_nodes=data.num_nodes,
                target_edge_index=data.test_edge_index,
                target_edge_type=data.test_edge_type,
                split="test",
            )
        dataset.data, dataset.slices = dataset.collate(
            [train_data, valid_data, test_data]
        )
    elif cls == "WN18RR":
        dataset = datasets.WordNet18RR(**dataset_cfg.dataset)
        # convert wn18rr into the same format as fb15k-237
        data = dataset.data
        num_nodes = int(data.edge_index.max()) + 1
        num_relations = int(data.edge_type.max()) + 1
        edge_index = data.edge_index[:, data.train_mask]
        edge_type = data.edge_type[data.train_mask]
        edge_index = torch.cat(
            [edge_index, edge_index.flip(0)], dim=-1
        )  # adding inverse edges
        edge_type = torch.cat(
            [edge_type, edge_type + num_relations]
        )  # adding inverse edge attr
        train_data = Data(
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=num_nodes,  # no masking is done
            target_edge_index=data.edge_index[:, data.train_mask],
            target_edge_type=data.edge_type[data.train_mask],
            split="train",
        )
        valid_data = Data(
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=num_nodes,
            target_edge_index=data.edge_index[:, data.val_mask],
            target_edge_type=data.edge_type[data.val_mask],
            split="valid",
        )
        if (
            combined_graph
        ):  # a graph that combines the train and valid edges for inference on test set
            valid_edge_index = torch.cat(
                (valid_data.target_edge_index, valid_data.target_edge_index.flip(0)),
                dim=1,
            )
            valid_edge_type = torch.cat(
                (
                    valid_data.target_edge_type,
                    valid_data.target_edge_type + num_relations,
                ),
                dim=0,
            )
            test_data = Data(
                edge_index=torch.cat((edge_index, valid_edge_index), dim=1),
                edge_type=torch.cat((edge_type, valid_edge_type), dim=0),
                num_nodes=num_nodes,
                target_edge_index=data.edge_index[:, data.test_mask],
                target_edge_type=data.edge_type[data.test_mask],
                split="test",
            )
        else:
            test_data = Data(
                edge_index=edge_index,
                edge_type=edge_type,
                num_nodes=num_nodes,
                target_edge_index=data.edge_index[:, data.test_mask],
                target_edge_type=data.edge_type[data.test_mask],
                split="test",
            )
        dataset.data, dataset.slices = dataset.collate(
            [train_data, valid_data, test_data]
        )
        dataset.num_relations = num_relations * 2

    elif cls in ["NELL995", "YAGO310"]:
        if cls == "NELL995":
            dataset = datasets.NELL995(**dataset_cfg.dataset)
        elif cls == "YAGO310":
            dataset = datasets.YAGO310(**dataset_cfg.dataset)
        data = dataset.data
        num_nodes = int(data.edge_index.max()) + 1
        num_relations = int(data.edge_type.max()) + 1
        dataset.num_relations = num_relations
        if (
            combined_graph
        ):  # a graph that combines the train and valid edges for inference on test set
            valid_edge_index = torch.cat(
                (dataset[1].target_edge_index, dataset[1].target_edge_index.flip(0)),
                dim=1,
            )
            valid_edge_type = torch.cat(
                (
                    dataset[1].target_edge_type,
                    dataset[1].target_edge_type + num_relations // 2,
                ),
                dim=0,
            )
            dataset[2].edge_index = torch.cat(
                (dataset[2].edge_index, valid_edge_index), dim=1
            )
            dataset[2].edge_type = torch.cat(
                (dataset[2].edge_type, valid_edge_type), dim=0
            )

    elif cls.startswith("Ind"):
        dataset = datasets.IndRelLinkPredDataset(name=cls[3:], **dataset_cfg.dataset)
    elif cls in ["aug_citation", "synthetic"]:
        dataset = hetero_datasets.HeteroDataset(name=cls, **dataset_cfg.dataset)

    else:
        raise ValueError("Unknown dataset `%s`" % cls)

    if hasattr(cfg, "topk_dir"):
        dataset.train_topk = torch.load(os.path.join(cfg.topk_dir, "train_topk.pt"))
        dataset.valid_topk = torch.load(os.path.join(cfg.topk_dir, "valid_topk.pt"))
        dataset.test_topk = torch.load(os.path.join(cfg.topk_dir, "test_topk.pt"))
        # filter out any duplicates
        for split in ["train", "valid", "test"]:
            split_topk = getattr(dataset, f"{split}_topk")
            all_query = split_topk[:, :2]
            topk_tails = split_topk[:, 2:]
            unique_query, unique_topk_tails = remove_duplicate(all_query, topk_tails)
            setattr(dataset, f"{split}_topk", (unique_query, unique_topk_tails))

    if hasattr(cfg.train, "train_on_expl") and cfg.train.train_on_expl:
        if hasattr(cfg.explainer_eval, "samples") and cfg.explainer_eval.samples:
            additional_info = ""
        else:
            ratio = getattr(cfg.explainer_eval, cfg.explainer_eval.eval_mask_type)
            additional_info = f"_{cfg.explainer_eval.eval_type}_{cfg.explainer_eval.eval_mask_type}_{ratio}"
        if cfg.train.fine_tune_on_val:
            splits = ["valid", "test"]
            split_ids = [1, 2]
        else:
            splits = ["train", "valid", "test"]
            split_ids = [0, 1, 2]
        for id, split in zip(split_ids, splits):
            expl = torch.load(
                os.path.join(
                    cfg.explainer_eval.expl_dir,
                    f"{split}_explanations{additional_info}.pt",
                ),
                map_location=torch.device("cpu"),
            )

            # filter out any duplicate queries
            all_query = expl[:, :2]
            explanations = expl[:, 2:]
            unique_query, unique_expl = remove_duplicate(all_query, explanations)

            # restrict the triplets to only those we have the explanations for.
            # each query must have its tail and head batch explanation
            target_query = torch.stack(
                (
                    dataset[id].target_edge_index[0],
                    dataset[id].target_edge_type,
                    dataset[id].target_edge_index[1],
                )
            ).T
            # create a unique number for each query by calculating entity_id * num_rel + rel_id
            tail_query_id = (
                dataset.num_relations * target_query[:, 0] + target_query[:, 1]
            )
            head_query_id = dataset.num_relations * target_query[:, 2] + (
                target_query[:, 1] + dataset.num_relations // 2
            )
            expl_query_id = (
                dataset.num_relations * unique_query[:, 0] + unique_query[:, 1]
            )
            tail_mask = torch.isin(tail_query_id, expl_query_id)
            head_mask = torch.isin(head_query_id, expl_query_id)
            mask = torch.logical_and(tail_mask, head_mask)
            target_edge_index = dataset[id].target_edge_index[:, mask]
            target_edge_type = dataset[id].target_edge_type[mask]

            setattr(
                dataset,
                f"{split}_target_triples",
                (target_edge_index, target_edge_type),
            )

            if hasattr(cfg.explainer_eval, "pad_expl") and cfg.explainer_eval.pad_expl:
                if get_rank() == 0:
                    # indicating which edges were selected
                    edge_mask = torch.zeros(
                        (unique_expl.size(0), dataset[0].edge_index.size(1)),
                        dtype=torch.bool,
                        device=unique_expl.device,
                    )
                    no_edge = unique_expl < 0
                    edge_id = unique_expl[~no_edge]
                    num_edges = (unique_expl >= 0).sum(1)
                    query_id = torch.arange(
                        unique_expl.size(0), device=unique_expl.device
                    ).repeat_interleave(num_edges)
                    edge_mask[query_id, edge_id] = True
                    rand_edges = torch.rand(edge_mask.shape)
                    rand_edges[edge_mask] = float("-inf")
                    budget = ratio - num_edges
                    rand_edges = rand_edges.argsort(descending=True, dim=1)
                    # get the selected padding edges
                    indices = torch.arange(rand_edges.size(1)).repeat(
                        unique_query.size(0), 1
                    )
                    selected_filter = indices < budget.unsqueeze(1)
                    # get the arg indices of the pad edge_id
                    arg_indices = rand_edges[selected_filter]
                    unique_expl[no_edge] = arg_indices.to(torch.int32)
                    # save the padded explanation
                    torch.save(
                        (unique_query, unique_expl),
                        os.path.join(
                            cfg.explainer_eval.expl_dir,
                            f"{split}_explanations{additional_info}_padded.pt",
                        ),
                    )
                synchronize()
                unique_query, unique_expl = torch.load(
                    os.path.join(
                        cfg.explainer_eval.expl_dir,
                        f"{split}_explanations{additional_info}_padded.pt",
                    ),
                )

            if hasattr(cfg.explainer_eval, "padding_expl_dir") and get_rank() == 0:
                # if there is padding required, load the padding edges.
                padding_expl = torch.load(
                    os.path.join(
                        cfg.explainer_eval.padding_expl_dir,
                        f"{split}_explanations{additional_info}.pt",
                    ),
                    map_location=torch.device("cpu"),
                )
                all_query = padding_expl[:, :2]
                explanations = padding_expl[:, 2:]
                pad_unique_query, pad_unique_expl = remove_duplicate(
                    all_query, explanations
                )
                assert torch.equal(unique_query, pad_unique_query)

                # mark the edge_id in pad_unique_expl that are duplicate to -1
                duplicate_mask = rowwise_isin(pad_unique_expl, unique_expl)
                pad_unique_expl[duplicate_mask] = -1
                # select the number pf edges to pad from the remaining ones.
                num_edges = (unique_expl >= 0).sum(1)
                max_budget = ratio - num_edges
                num_padding_edges = (pad_unique_expl >= 0).sum(1)
                budget = torch.min(num_padding_edges, max_budget)
                # shuffle the padding edges, but put any padding tokens (-1) to be last (and therefore not selected)
                pad_unique_expl = pad_unique_expl.sort(descending=True, dim=1)[0]
                rand = torch.rand(pad_unique_expl.shape)
                rand[pad_unique_expl < 0] = float("-inf")
                rand = rand.argsort(descending=True, dim=1)
                # get the selected padding edges
                indices = torch.arange(ratio).repeat(unique_query.size(0), 1)
                selected_filter = indices < budget.unsqueeze(1)
                # get the arg indices of the pad edge_id
                arg_indices = rand[selected_filter]
                batch_id = torch.arange(unique_query.size(0)).repeat_interleave(budget)
                # get the pad edge_id
                pad_edge_id = pad_unique_expl[batch_id, arg_indices]
                # add the padding edges to the expl
                selected_filter = torch.logical_and(
                    indices >= num_edges.unsqueeze(1),
                    indices < (budget.unsqueeze(1) + num_edges.unsqueeze(1)),
                )
                unique_expl[selected_filter] = pad_edge_id

                # save the padded explanation
                torch.save(
                    (unique_query, unique_expl),
                    os.path.join(
                        cfg.explainer_eval.expl_dir,
                        f"{split}_explanations{additional_info}_padded.pt",
                    ),
                )

            if hasattr(cfg.explainer_eval, "padding_expl_dir") and get_rank() != 0:
                # wait until rank 0 processes the data
                synchronize()
                unique_query, unique_expl = torch.load(
                    os.path.join(
                        cfg.explainer_eval.expl_dir,
                        f"{split}_explanations{additional_info}_padded.pt",
                    )
                )

            setattr(dataset, f"{split}_expl", (unique_query, unique_expl))
        # wait until all processes are complete
        synchronize()

    if get_rank() == 0:
        logger.warning("%s dataset" % cls)
        if hasattr(dataset, "train_target_triples"):
            train_num_edges = dataset.train_target_triples[0].shape[1]
        else:
            train_num_edges = dataset[0].target_edge_index.shape[1]
        if hasattr(dataset, "valid_target_triples"):
            valid_num_edges = dataset.valid_target_triples[0].shape[1]
            test_num_edges = dataset.test_target_triples[0].shape[1]
        else:
            valid_num_edges = dataset[1].target_edge_index.shape[1]
            test_num_edges = dataset[2].target_edge_index.shape[1]
        logger.warning(
            "#train: %d, #valid: %d, #test: %d"
            % (train_num_edges, valid_num_edges, test_num_edges)
        )

    return dataset


def build_model(cfg):
    model_cfg = copy.deepcopy(cfg)
    cls = model_cfg.model.pop("class")
    if cls == "NBFNet":
        model = models.NBFNet(**model_cfg.model)
    elif cls == "RGCN":
        model = rgcn.RGCN(**model_cfg.model)
    else:
        raise ValueError("Unknown model `%s`" % cls)
    if "checkpoint" in model_cfg:
        state = torch.load(model_cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])

    return model


def get_model(model_cfg, cls):
    if cls == "NBFNet":
        model = models_expl.NBFNet(**model_cfg.model)
    elif cls == "RGCN":
        model = rgcn.RGCN(**model_cfg.model)
    else:
        raise ValueError("Unknown model `%s`" % cls)
    return model


def build_model_expl(cfg):
    model_cfg = copy.deepcopy(cfg)
    cls = model_cfg.model.pop("class")

    model = get_model(model_cfg, cls)

    if "checkpoint" in model_cfg:
        state = torch.load(model_cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])

    if "eval_checkpoint" in model_cfg:
        eval_model = get_model(model_cfg, cls)
        state = torch.load(model_cfg.eval_checkpoint, map_location="cpu")
        eval_model.load_state_dict(state["model"])
    else:
        eval_model = None

    return model, eval_model


def build_explainer_dataset(cls, root, dataset):
    dataset = datasets.ExplanationDataset(root=root, name=cls, dataset=dataset, hops=6)
    return dataset


def load_id2name(dataset, cls):
    if cls == "FB15k-237":
        with open(os.path.join(dataset.raw_dir, "entities.dict")) as f:
            lines = [row.split("\t") for row in f.read().split("\n")[:-1]]
            entity2id = {key: int(value) for value, key in lines}

        with open(os.path.join(dataset.raw_dir, "relations.dict")) as f:
            lines = [row.split("\t") for row in f.read().split("\n")[:-1]]
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

    id2entity = {}
    id2relation = {}
    for k, v in entity2id.items():
        id2entity[v] = k
    for k, v in relation2id.items():
        id2relation[v] = k

    return id2entity, id2relation


def wandb_setup(cfg, rank):
    if cfg.wandb.use and rank == 0:
        try:
            import wandb
        except:
            raise ImportError("WandB is not installed.")
        wandb_name = cfg.wandb.name
        run = wandb.init(
            entity=cfg.wandb.entity, project=cfg.wandb.project, name=wandb_name
        )
        run.config.update(cfg)
    else:
        run = None
    return run


def save_model(logger, epoch, model, optimizer):
    logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, "model_epoch_%d.pth" % epoch)
