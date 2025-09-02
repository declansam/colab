import os
import sys
import math
import pprint
import copy
import pandas as pd
from tqdm import tqdm
import numpy as np
import time
from collections import defaultdict
import torch
from torch import optim
from torch import nn
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import tasks, util
from explainers import data_util, explainer_util

separator = ">" * 30
line = "-" * 30


@torch.no_grad()
def test(
    cfg,
    model,
    test_data,
    logger,
):
    world_size = util.get_world_size()
    rank = util.get_rank()

    # * Data Loading *
    test_triplets = torch.cat(
        [test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]
    ).t()
    # test_triplets = test_triplets[:30]
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(
        test_triplets, cfg.train.batch_size, sampler=sampler
    )

    # * Logging Setup *
    count = []

    # * Miscellaneous *
    if rank == 0:
        pbar = tqdm(total=len(test_loader))

    assert cfg.train.train_on_expl
    test_on_subgraph = True

    # * Test *
    for batch in test_loader:
        # * Triple & Filtering Setup*
        batch_size = len(batch)
        t_batch, h_batch = tasks.all_negative(test_data, batch)

        # * Tail Prediction *
        central_nodes = batch[:, 0].unsqueeze(1)

        # * Prepare the data *
        if test_on_subgraph:
            # create dynamic graphs per query so that the negative samples are sampled from the subgraphs.
            if cfg.train.train_on_expl:
                # prepare the data based on the explanation
                c = data_util.prepare_expl_data(
                    cfg,
                    test_data,
                    central_nodes,
                    t_batch,
                    train_on_counter_factual=cfg.train.train_on_counter_factual,
                    count_components=True,
                    ignore_center=True,
                )

        count.append(c)

        # * Head Prediction *
        mode = torch.zeros((batch_size,), dtype=torch.bool, device=batch.device)
        h_batch = tasks.conversion_to_tail_prediction(h_batch, model.num_relation, mode)
        central_nodes = batch[:, 1].unsqueeze(1)

        # * Prepare the data *
        if test_on_subgraph:
            # create dynamic graphs per query so that the negative samples are sampled from the subgraphs.
            if cfg.train.train_on_expl:
                # prepare the data based on the explanation
                c = data_util.prepare_expl_data(
                    cfg,
                    test_data,
                    central_nodes,
                    h_batch,
                    train_on_counter_factual=cfg.train.train_on_counter_factual,
                    count_components=True,
                    ignore_center=True,
                )
        count.append(c)

        if rank == 0:
            pbar.update(1)
    if rank == 0:
        pbar.close()

    count = torch.tensor(count)
    logger.warning(f"Average Number of Connected Components: {count.mean():.2f}")

    return


def run(cfg, args, working_dir, logger, run_id=0, num_runs=1, log_separate=False):
    if num_runs > 1 or log_separate:
        if log_separate:
            run_id = f"{cfg.model['class']}_{run_id}"
        working_dir = util.create_working_directory_per_run(run_id)
        logger = util.change_logger_file(logger)

    torch.manual_seed(args.seed + util.get_rank())
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(f"Process ID: {os.getpid()}")
        logger.warning(pprint.pformat(cfg))

    # * Dataset Setup *
    is_inductive = cfg.dataset["class"].startswith("Ind")
    dataset = util.build_dataset(cfg)

    # * Model Setup *
    cfg.model.num_relation = dataset.num_relations
    model, _ = util.build_model_expl(cfg)

    # * Send to Device *
    device = util.get_device(cfg)
    model = model.to(device)
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]

    if cfg.train.train_on_expl:
        if not cfg.train.fine_tune_on_val:
            train_data.expl = dataset.train_expl
        valid_data.expl = dataset.valid_expl
        test_data.expl = dataset.test_expl
        if hasattr(dataset, "train_target_triples"):
            train_data.target_edge_index, train_data.target_edge_type = (
                dataset.train_target_triples[0],
                dataset.train_target_triples[1],
            )
        if hasattr(dataset, "valid_target_triples"):
            valid_data.target_edge_index, valid_data.target_edge_type = (
                dataset.valid_target_triples[0],
                dataset.valid_target_triples[1],
            )
            test_data.target_edge_index, test_data.target_edge_type = (
                dataset.test_target_triples[0],
                dataset.test_target_triples[1],
            )

    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)

    # * Standard Evaluation *
    for split_data, split_name in zip([test_data], ["test"]):
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning(f"Evaluate on {split_name}")
        test(
            cfg,
            model,
            split_data,
            logger,
        )

    return


if __name__ == "__main__":
    # * Cfg, Dir, Logger Setup *
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)
    logger = util.get_root_logger()
    avg_split_score = {}
    # Get the value for X runs
    for i in range(args.num_runs):
        # change seed every run
        args.seed += i
        run(cfg, args, working_dir, logger, run_id=i, num_runs=args.num_runs)
        if args.num_runs > 1:
            # set the dir back to the original in case we have a run-specific dir.
            os.chdir(working_dir)
            logger = util.change_logger_file(logger)
