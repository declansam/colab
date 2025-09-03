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


def get_topk(
    batch,
    pred,
):
    """
    Prepares the topk tails. The first two columns will be (head, rel).
    The next columns will be [tail_id_0, ..., tail_id_k] for the topk tails.
    """
    h_index, t_index, r_index = batch.unbind(-1)  # (batch_size, num_triples)
    # assumption: the batch has been converted to tail batch
    assert (h_index[:, [0]] == h_index).all()
    assert (r_index[:, [0]] == r_index).all()
    # prepare the query (h, r)
    query = batch[:, 0, :]
    query = query[:, [0, 2]].to("cpu")
    tails = torch.argsort(pred, dim=1, descending=True)
    topk_tails = tails[:, : cfg.eval.save_topk].to("cpu")
    topk_tails = torch.cat((query, topk_tails), dim=1)
    return topk_tails


def train_and_validate(
    cfg,
    model,
    train_data,
    valid_data,
    logger,
    device,
    filtered_data=None,
    randomwalker=None,
):
    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()

    # * Wanbd Settings *
    run = util.wandb_setup(cfg, rank)

    # * Data Loading *
    '''
    # Example with 5 training triples
    train_data.target_edge_index = torch.tensor([
        [0, 1, 2, 3, 4],  # Head entities (source nodes)
        [5, 6, 7, 8, 9]   # Tail entities (target nodes)
    ])

    train_data.target_edge_type = torch.tensor([0, 1, 2, 0, 1])  # Relation types

    so results:
    torch.tensor([
        [0, 1, 2, 3, 4],  # Head entities
        [5, 6, 7, 8, 9],  # Tail entities  
        [0, 1, 2, 0, 1]   # Relations
    ])
    '''

    # [num_edges, 3]
    train_triplets = torch.cat(
        [train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0)]
    ).t()

    # train_triplets = train_triplets[:30]
    # dividing data portion among GPUs
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
    train_loader = torch_data.DataLoader(
        train_triplets, cfg.train.batch_size, sampler=sampler
    )

    # * Optimizer *
    optim_cfg = copy.deepcopy(cfg)
    cls = optim_cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(model.parameters(), **optim_cfg.optimizer)

    # * DDP Setup *
    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    # * Best Model Selection *
    best_result = float("-inf")
    best_epoch = -1

    # * Miscellaneous *
    #$ Determine step = frequency of validation
    if hasattr(cfg.train, "step"):
        step = cfg.train.step
    else:
        step = math.ceil(cfg.train.num_epoch / 10)
    is_synthetic = cfg.dataset["class"] in ["synthetic", "aug_citation"]
    batch_id = 0

    if cfg.train.train_on_expl or model.distance_dropout:
        train_on_subgraph = True
    else:
        train_on_subgraph = False

    # * Training Loop *
    for epoch in range(0, cfg.train.num_epoch):
        # * Epoch Setup *
        parallel_model.train()
        sampler.set_epoch(epoch)
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning("Epoch %d begin" % epoch)
            pbar = tqdm(total=len(train_loader), desc="Processing Batch")
        losses = []

        for batch in train_loader:
            
            # * Mode Setup *
            #$ Determine the mode for prediction (head or tail)
            #$ mode[i] = 1 means tail prediction, 0 means head prediction
            #$ tail prediction; (h, r, ?)
            batch_size = len(batch)
            if is_synthetic:
                # for synthetic data, we do tail batch only
                mode = torch.ones((batch_size,), dtype=torch.bool, device=batch.device)

            #$ head prediction; (?, r, t)
            #$ start by 0 i.e. head prediction for all
            #$ BUT, modify the first half to be tail prediction by setting them to 1
            #$ This mixed mode is used for balanced training
            else:
                mode = torch.zeros((batch_size,), dtype=torch.bool, device=batch.device)
                # first half will be tail batch, and the second half will be head batch
                mode[: batch_size // 2] = 1
            #! mode = [batch_size]

            # * Data Setup *
            # create the central nodes (could be used for RW)
            #$ pick batch[:, 0] if mode = 1, else pick batch[:, 1]
            #$ heads signify the nodes to be used for RW
            #! heads = [batch_size], central_nodes = [batch_size, 1]
            heads = torch.where(mode, batch[:, 0], batch[:, 1])
            central_nodes = heads.unsqueeze(1)


            '''
            Query 0 (Tail Prediction):
                Central Node: 0 (dog)
                Task: (dog, hypernym, ?) → predict cat

                    0 (dog) ──hypernym──> ? 
                    ↑
                Central Node (starting point for GNN)

            Query 4 (Head Prediction):  
                Central Node: 9 (mammal)
                Task: (?, hypernym, mammal) → predict lion

                    ? ──hypernym──> 9 (mammal)
                                    ↑
                                Central Node (starting point for GNN)
            '''

            # start = time.time()
            if train_on_subgraph:
                # create dynamic graphs per query so that the negative samples are sampled from the subgraphs.
                # convert the batch to tail prediction to obtain the query
                query_batch = batch.unsqueeze(1)
                query_batch = tasks.conversion_to_tail_prediction(
                    query_batch, model.num_relation, mode
                )
                if cfg.train.train_on_expl:
                    # prepare the data based on the explanation
                    data = data_util.prepare_expl_data(
                        cfg,
                        train_data,
                        central_nodes,
                        query_batch,
                        train_on_counter_factual=cfg.train.train_on_counter_factual,
                    )
                else:
                    data = data_util.prepare_dist_data(
                        train_data,
                        central_nodes,
                        hops=model.max_dropout_distance,
                        eval_on_edge_drop=model.eval_on_edge_drop,
                        randomized_edge_drop=model.randomized_edge_drop,
                        eval_dropout_distance=model.eval_dropout_distance,
                        max_edge_drop_prob=model.max_edge_drop_prob,
                    )

                # get the negative sample from the ego_network
                batch, pos_included = tasks.negative_sampling_from_ego(
                    train_data,
                    data,
                    batch,
                    cfg.task.num_negative,
                    mode,
                    strict=cfg.task.strict_negative,
                    is_synthetic=is_synthetic,
                )
                # convert the batch to tail prediction
                batch = tasks.conversion_to_tail_prediction(
                    batch, model.num_relation, mode
                )
                # relabel the batch and also get the mask indicating the tails that are inside the ego network
                batch, valid_tails = data_util.relabel_batch(data, batch)

            else:
                # create negative samples
                # batch prev: [batch_size, 3]
                # batch now:  [batch_size, num_negatives + 1, 3]  (num_negatives + 1 because of the positive sample)
                batch = tasks.negative_sampling(
                    train_data,
                    batch,
                    cfg.task.num_negative,
                    strict=cfg.task.strict_negative,
                    is_synthetic=is_synthetic,
                    mode=mode,
                )

                # convert the batch to tail prediction
                # (h, t, r) -> (t, r, r') where r' = r + num_relation // 2 (since original tail prediction relation are unchanged)
                # shape still: [batch_size, num_negatives + 1, 3]
                batch = tasks.conversion_to_tail_prediction(
                    batch, model.num_relation, mode
                )

                # Graph data replicated for batch processing
                # Take my full graph, make batch_size identical copies, and keep track of which node/edge belongs to which copy.
                # Returns Data() object that contains:
                # - The original graph replicated batch_size times.
                # - Which nodes belong to which batch copy.
                # - Which edges belong to which batch copy.
                # - Central nodes for each copy.
                # Shape stays the same for each copy - we don’t shrink or expand the graph, we just repeat it.
                '''
                INPUT:
                    edge_index = [[0,1,2],
                                [1,2,0]]        # [2, num_edges=3]
                    edge_type  = [0,1,2]          # [num_edges=3]
                    num_nodes  = 3
                    batch_size = 2

                OUTPUT:
                    s_data.edge_index =
                    [[0,1,2,0,1,2],
                    [1,2,0,1,2,0]]               # [2, num_edges * batch_size] = [2,6]

                    s_data.edge_type =
                    [0,1,2,0,1,2]                # [6]

                    s_data.node_id =
                    [0,1,2,0,1,2]                # [num_nodes * batch_size] = [6]

                    s_data.node_batch =
                    [0,0,0,1,1,1]                # each node belongs to graph 0 or 1

                    s_data.edge_batch =
                    [0,0,0,1,1,1]                # each edge belongs to graph 0 or 1

                    s_data.subgraph_num_nodes =
                    [3,3]                        # each graph copy has 3 nodes

                    s_data.subgraph_num_edges =
                    [3,3]                        # each graph copy has 3 edges
                '''
                data = data_util.prepare_full_data(
                    train_data,
                    batch_size,
                    central_nodes,
                    hops=model.max_dropout_distance,
                    distance_dropout=model.distance_dropout,
                )

            # Batched graph data optimized for parallel processing
            # Rearranges the graph data so that NBFNet can process different graphs at the same time.
            '''
            From above, s_data has:
                edge_index = [2, 6] (flat)
                edge_batch = [0,0,0,1,1,1]
                subgraph_num_edges = [3,3]

            OUTPUT:
            data.batched_edge_index =
                                        [[[0,1,2],
                                        [0,1,2]],         # shape [2, batch_size=2, max_num_edges=3]
                                        [[1,2,0],
                                        [1,2,0]]]

                                        data.batched_edge_type =
                                        [[0,1,2],
                                        [0,1,2]]           # [batch_size=2, max_num_edges=3]

                                        data.edge_filter =
                                        [[1,1,1],
                                        [1,1,1]]           # 1 = valid edge, 0 = padded edge

            '''
            data = data_util.create_batched_data(data)  # create the batched data

            # do masking, random walk if applicable
            # Before masking: Graph has all original edges including direct paths from heads to tails
            # After masking: Direct edges like (head_0, relation_0, tail_0) are removed from the graph
            #                Model must find alternative paths like head_0 → intermediate_node → tail_0
            #                Forces learning of compositional reasoning patterns
            # Insight: The masking doesn't change your batch tensor [16, 33, 3]
            # it modifies the underlying graph structure that the model will use to score those 16×33 candidate triples.
            data = model.masking(data, batch, randomwalker)
            # end = time.time()
            # if util.get_rank() == 0:
            #     logger.warning(f"* Data Prep Took {(end-start):.2f}s*")

            # * Forward, Loss, Backprop*
            pred = parallel_model(data, batch)

            # if util.get_rank() == 0:
            #     logger.warning(f"* Forward Prep Took {(time.time() - end):.2f}s*")

            target = torch.zeros_like(pred)
            target[:, 0] = 1
            if train_on_subgraph:
                target = target.to(torch.bool)
                loss = explainer_util.compute_loss_ego(
                    cfg, pred, target, pos_included, valid_tails
                )
            else:
                loss = explainer_util.compute_loss(cfg, pred, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if util.get_rank() == 0:
                pbar.update(1)
                if batch_id % cfg.train.log_interval == 0:
                    logger.warning(separator)
                    logger.warning("binary cross entropy: %g" % loss)
            losses.append(loss.item())
            batch_id += 1

        # * Epoch Log *
        if util.get_rank() == 0:
            avg_loss = sum(losses) / len(losses)
            logger.warning(separator)
            logger.warning("Epoch %d end" % epoch)
            logger.warning(line)
            logger.warning("average binary cross entropy: %g" % avg_loss)
            pbar.close()

            # * Wandb Log *
            if cfg.wandb.use:
                if epoch % step == 0 or epoch == cfg.train.num_epoch - 1:
                    commit = False
                else:
                    commit = True
                stats = {"train/loss": avg_loss}
                run.log(stats, step=epoch, commit=commit)

        util.synchronize()

        # * Model Validation *
        if epoch % step == 0 or epoch == cfg.train.num_epoch - 1:
            # * Validate Model *
            if hasattr(cfg.train, "select_using_dist") and cfg.train.select_using_dist:
                # evaluate performance on distance edge drop

                # record the orig performance
                eval_on_edge_drop = model.eval_on_edge_drop
                distance_dropout = model.distance_dropout
                randomized_edge_drop = model.randomized_edge_drop
                distance = model.eval_dropout_distance

                _, mrr = test_distance(
                    cfg,
                    model,
                    valid_data,
                    "valid",
                    logger,
                    device,
                    filtered_data,
                    run=run,
                    epoch=epoch,
                )

                # revert back the settings.
                model.eval_on_edge_drop = eval_on_edge_drop
                model.distance_dropout = distance_dropout
                model.randomized_edge_drop = randomized_edge_drop
                model.eval_dropout_distance = distance

                # calculate the avg mrr
                result = sum(mrr) / len(mrr)

            else:
                if rank == 0:
                    logger.warning(separator)
                    logger.warning("Evaluate on valid")
                result, _ = test(
                    cfg,
                    model,
                    valid_data,
                    logger,
                    device,
                    filtered_data=filtered_data,
                    split="valid",
                    run=run,
                    epoch=epoch,
                )

            # * Save Model *
            if result > best_result:
                best_result = result
                best_epoch = epoch
                if rank == 0:
                    logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
                    state = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    torch.save(state, "model_epoch_%d.pth" % epoch)
        util.synchronize()

    # * Load Best Model *
    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device)
    model.load_state_dict(state["model"])
    util.synchronize()


@torch.no_grad()
def test(
    cfg,
    model,
    test_data,
    logger,
    device,
    filtered_data=None,
    groundtruth_data=None,
    working_dir=None,
    split="test",
    run=None,
    epoch=None,
    commit=True,
    final_evaluation=False,
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

    # * Model Setup *
    model.eval()

    # * Logging Setup *
    rankings = []
    num_negatives = []
    modes = []  # logging the mode (1 for tail, 0 for head)
    heads = []
    rels = []  # logging the rel types.
    tails = []

    # * Miscellaneous *
    is_synthetic = cfg.dataset["class"] in ["synthetic", "aug_citation"]
    if rank == 0:
        pbar = tqdm(total=len(test_loader))

    if cfg.eval.save_topk > 0:
        save_topk = True
        topk = []
        assert world_size == 1
    else:
        save_topk = False
        topk = []

    if cfg.train.train_on_expl or (model.distance_dropout and model.eval_on_edge_drop):
        test_on_subgraph = True
    else:
        test_on_subgraph = False

    # * Test *
    for batch in test_loader:
        # * Triple & Filtering Setup*
        batch_size = len(batch)
        t_batch, h_batch = tasks.all_negative(test_data, batch)
        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()

        # * Tail Prediction *
        central_nodes = batch[:, 0].unsqueeze(1)

        # * Prepare the data *
        if test_on_subgraph:
            # create dynamic graphs per query so that the negative samples are sampled from the subgraphs.
            if cfg.train.train_on_expl:
                # prepare the data based on the explanation
                data = data_util.prepare_expl_data(
                    cfg,
                    test_data,
                    central_nodes,
                    t_batch,
                    train_on_counter_factual=cfg.train.train_on_counter_factual,
                )
            else:
                data = data_util.prepare_dist_data(
                    test_data,
                    central_nodes,
                    hops=model.max_dropout_distance,
                    eval_on_edge_drop=model.eval_on_edge_drop,
                    randomized_edge_drop=model.randomized_edge_drop,
                    eval_dropout_distance=model.eval_dropout_distance,
                    max_edge_drop_prob=model.max_edge_drop_prob,
                )

            t_batch = data_util.check_if_tail_in_network(data, t_batch)
            t_batch, t_valid_tails = data_util.relabel_batch(data, t_batch)
        else:
            data = data_util.prepare_full_data(
                test_data,
                batch_size,
                central_nodes,
                hops=model.max_dropout_distance,
                distance_dropout=model.distance_dropout,
            )
            data = data_util.get_groundtruth(data, groundtruth_data, batch[:, :2])

        data = data_util.create_batched_data(data)  # create the batched data
        # do masking
        data = model.masking(data, t_batch)  # do the random walk
        t_pred = model(data, t_batch)  # t_batch: (batch_size, num_nodes, 3)

        if test_on_subgraph:
            # for the predictions outside the subgraph, it will get a score of -inf
            t_pred[~t_valid_tails] = float("-inf")

        # * Rank and Stat Computation *
        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        num_t_negative = t_mask.sum(dim=-1)

        if save_topk:
            topk_tails = get_topk(t_batch, t_pred)
            topk.append(topk_tails)

        # * Tail-Prediction only for Synthetic*
        if is_synthetic:
            rankings += [t_ranking]
            num_negatives += [num_t_negative]
            heads += [batch[:, 0]]
            tails += [batch[:, 1]]
            rels += [batch[:, -1]]
            modes += [torch.ones(t_ranking.shape)]
            if rank == 0:
                pbar.update(1)
            continue

        # * Head Prediction *
        mode = torch.zeros((batch_size,), dtype=torch.bool, device=batch.device)
        h_batch = tasks.conversion_to_tail_prediction(h_batch, model.num_relation, mode)
        central_nodes = batch[:, 1].unsqueeze(1)

        # * Prepare the data *
        if test_on_subgraph:
            # create dynamic graphs per query so that the negative samples are sampled from the subgraphs.
            if cfg.train.train_on_expl:
                # prepare the data based on the explanation
                data = data_util.prepare_expl_data(
                    cfg,
                    test_data,
                    central_nodes,
                    h_batch,
                    train_on_counter_factual=cfg.train.train_on_counter_factual,
                )
            else:
                data = data_util.prepare_dist_data(
                    test_data,
                    central_nodes,
                    hops=model.max_dropout_distance,
                    eval_on_edge_drop=model.eval_on_edge_drop,
                    randomized_edge_drop=model.randomized_edge_drop,
                    eval_dropout_distance=model.eval_dropout_distance,
                    max_edge_drop_prob=model.max_edge_drop_prob,
                )

            h_batch = data_util.check_if_tail_in_network(data, h_batch)
            h_batch, h_valid_tails = data_util.relabel_batch(data, h_batch)
        else:
            data = data_util.prepare_full_data(
                test_data,
                batch_size,
                central_nodes,
                hops=model.max_dropout_distance,
                distance_dropout=model.distance_dropout,
            )
        data = data_util.create_batched_data(data)  # create the batched data
        # do masking
        data = model.masking(data, h_batch)  # do the random walk
        h_pred = model(data, h_batch)

        if test_on_subgraph:
            h_pred[~h_valid_tails] = float("-inf")

        # * Rank and Stat Computation *
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
        num_h_negative = h_mask.sum(dim=-1)

        if save_topk:
            topk_tails = get_topk(h_batch, h_pred)
            topk.append(topk_tails)

        # * Logging *
        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]
        heads += [batch[:, 0], batch[:, 1]]
        tails += [batch[:, 1], batch[:, 0]]
        rels += [
            batch[:, -1],
            batch[:, -1],
        ]  # get the rels for both tail and head predictions
        modes += [
            torch.ones(t_ranking.shape),
            torch.zeros(h_ranking.shape),
        ]  # log whether it was for tail prediction or head prediction
        if rank == 0:
            pbar.update(1)
    if rank == 0:
        pbar.close()

    # * Aggr statistics across devices *
    stats = {
        "ranking": torch.cat(rankings),
        "num_negative": torch.cat(num_negatives),
        "heads": torch.cat(heads),
        "tails": torch.cat(tails),
        "rels": torch.cat(rels),
        "modes": torch.cat(modes),
    }
    # * Save Topk *
    if save_topk:
        topk = torch.cat(topk)
    all_stats, _ = explainer_util.combine_stats(rank, world_size, device, stats)
    # * Calculate Metric *
    _, scores = explainer_util.calc_metric_and_save_result(
        cfg,
        rank,
        eval_type=None,
        eval_mask_type=None,
        ratio=None,
        all_stats=all_stats,
        split=split,
        run=run,
        epoch=epoch,
        commit=commit,
        working_dir=working_dir,
        save_topk=save_topk,
        topk=topk,
        final_evaluation=final_evaluation,
        distance=model.eval_dropout_distance,
    )

    mrr = (1 / all_stats["ranking"].float()).mean()
    return mrr, scores


def synthetic_evaluation(
    dataset, model, test_data, logger, device, filtered_data, working_dir
):
    """
    Does model evaluation on synthetic dataset where the ground truth pattern is present.
    1. Experiment on evaluating the performance when the ground truth is removed.
    This experiment compares the treatment (removing the ground truth) to the control (removing random edges)
    2. Experiment on evaluating the GNN performance when ground truth is kept but changing noise level.
    This experiment compares the treatment (removing noise) to the control (removing random noise)
    """
    groundtruth_data = dataset.groundtruth_data
    groundtruth_data = groundtruth_data.to(device)

    # evaluate performance when ground truth is removed
    if cfg.eval.remove_ground_truth:
        # Treatment: Removing the Ground Truth
        model.remove_ground_truth = True
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning(f"Evaluate on test with ground truth removed")
        test(
            cfg,
            model,
            test_data,
            logger,
            device,
            filtered_data=filtered_data,
            groundtruth_data=groundtruth_data,
            working_dir=working_dir,
            split="test",
        )
        model.remove_ground_truth = False

        # Control: Removing the same number of random edges that is not the Ground Truth
        model.remove_ground_truth_control = True
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning(f"Evaluate on test with ground truth removed CONTROL")
        test(
            cfg,
            model,
            test_data,
            logger,
            device,
            filtered_data=filtered_data,
            groundtruth_data=groundtruth_data,
            working_dir=working_dir,
            split="test",
        )
        model.remove_ground_truth_control = False

    # evaluate performance when ground truth is kept but changing noise level
    if cfg.eval.keep_ground_truth:
        model.keep_ground_truth = True
        model.eval_on_edge_drop = True
        for randomized_edge_drop in cfg.eval.randomized_edge_drop:
            model.randomized_edge_drop = randomized_edge_drop
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning(
                    f"Evaluate on test keeping the ground truth with randomized edge drop {randomized_edge_drop}"
                )
            test(
                cfg,
                model,
                test_data,
                logger,
                device,
                filtered_data=filtered_data,
                groundtruth_data=groundtruth_data,
                working_dir=working_dir,
                split="test",
            )
        # reset the randomized drop edge prob
        model.keep_ground_truth = False
        model.eval_on_edge_drop = False
        model.randomized_edge_drop = 0

        model.keep_ground_truth_control = True
        model.eval_on_edge_drop = True
        for randomized_edge_drop in cfg.eval.randomized_edge_drop:
            model.randomized_edge_drop = randomized_edge_drop
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning(
                    f"Evaluate on test keeping the ground truth with randomized edge drop {randomized_edge_drop} CONTROL"
                )
            test(
                cfg,
                model,
                test_data,
                logger,
                device,
                filtered_data=filtered_data,
                groundtruth_data=groundtruth_data,
                working_dir=working_dir,
                split="test",
            )
        # reset the randomized drop edge prob
        model.keep_ground_truth_control = False
        model.eval_on_edge_drop = False
        model.randomized_edge_drop = 0

    return


def eval_on_edge_drop(
    model, split_data, split_name, logger, device, filtered_data, working_dir
):
    """
    Evaluate performance on varying edge drop probability
    """
    model.eval_on_edge_drop = True
    # disable model distance dropout
    model.distance_dropout = False
    model.eval_dropout_distance = -1
    all_scores = {}
    for randomized_edge_drop in cfg.eval.randomized_edge_drop:
        model.randomized_edge_drop = randomized_edge_drop
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning(
                f"Evaluate on {split_name} with randomized edge drop {randomized_edge_drop}"
            )
        _, scores = test(
            cfg,
            model,
            split_data,
            logger,
            device,
            filtered_data=filtered_data,
            working_dir=working_dir,
            split=split_name,
        )
        all_scores[randomized_edge_drop] = scores

    # save the score as a dataframe
    if util.get_rank() == 0:
        df = pd.DataFrame(all_scores)
        df.to_csv(os.path.join(working_dir, f"{split_name}_eval_on_edge_drop.csv"))

    # reset the randomized drop edge prob
    model.eval_on_edge_drop = False
    model.randomized_edge_drop = 0


def test_distance(
    cfg,
    model,
    split_data,
    split_name,
    logger,
    device,
    filtered_data,
    working_dir=None,
    run=None,
    epoch=None,
    commit=True,
):
    model.eval_on_edge_drop = True
    model.distance_dropout = True
    model.randomized_edge_drop = 0
    all_scores = {}
    if hasattr(cfg.eval, "eval_distance"):
        distances = cfg.eval.eval_distance
    else:
        distances = range(1, cfg.model.max_dropout_distance + 1)
    mrr = []
    for i, distance in enumerate(distances):
        model.eval_dropout_distance = distance
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning(
                f"Evaluate on {split_name} with distance edge drop {distance}"
            )
        if commit and i == len(distances) - 1:
            commit_flag = True
        else:
            commit_flag = False
        result, scores = test(
            cfg,
            model,
            split_data,
            logger,
            device,
            filtered_data=filtered_data,
            split=split_name,
            run=run,
            epoch=epoch,
            working_dir=working_dir,
            commit=commit_flag,
        )
        all_scores[distance] = scores
        mrr.append(result)
    return all_scores, mrr


def eval_on_distance_drop(
    cfg, model, split_data, split_name, logger, device, filtered_data, working_dir
):
    """
    Evaluate performance on varying distance edge drop
    """
    all_scores, _ = test_distance(
        cfg, model, split_data, split_name, logger, device, filtered_data, working_dir
    )

    if util.get_rank() == 0:
        df = pd.DataFrame(all_scores)
        df.to_csv(os.path.join(working_dir, f"{split_name}_eval_on_distance_drop.csv"))

    # reset the randomized drop edge prob
    model.eval_on_edge_drop = False
    model.distance_dropout = False
    model.eval_dropout_distance = -1


def run(cfg, args, working_dir, logger, run_id=0, num_runs=1, log_separate=False):

    # If multiple runs, create separate directories for each run
    if num_runs > 1 or log_separate:
        if log_separate:
            run_id = f"{cfg.model['class']}_{run_id}"
        working_dir = util.create_working_directory_per_run(run_id)
        logger = util.change_logger_file(logger)

    # Set random seed (different for distributed processes)
    torch.manual_seed(args.seed + util.get_rank())

    # Print experiment info (seed, config, process ID) for rank 0
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(f"Process ID: {os.getpid()}")
        logger.warning(pprint.pformat(cfg))

    # * Dataset Setup *
    # Inductive vs Transductive: Determines if we're learning on unseen entities
    #   - Transductive: Same entities in train/test (e.g., WN18RR)
    #   - Inductive: New entities in test (e.g., FB15K-237)
    #$ here, dataset is a LIST of three objects (test_data, train_data, valid_data), 
    #$ each of which is an object for specific name such as WordNet18RR()
    #$ dataset has (among others): 
    ##     - Data object, edge2id, num_classes, num_features, ..., raw/processed dir info, edge_index, edge_type, etc.
    ##     - test_data, train_data, valid_data, train/test/valid_num_edges
    is_inductive = cfg.dataset["class"].startswith("Ind")
    dataset = util.build_dataset(cfg)

    # * Model Setup *
    # get the number of relations from the dataset
    # build the model (NBFNet or NBFNetExpl)
    #$ differene between build_model and build_model_expl: build_model_expl is for explanation, build_model is for training
    #$ ie build_model_expl returns both model and eval_model BUT build_model returns only model
    cfg.model.num_relation = dataset.num_relations
    model, _ = util.build_model_expl(cfg)

    # * Send to Device *
    device = util.get_device(cfg)
    model = model.to(device)
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]

    # Explainers (train_on_expl: False in the config file config/nbfnet/wn18rr-0b.yaml)
    # instead of training on the full knowledge graph, the model trains on relevant subgraphs that explain why certain facts are true.
    if cfg.train.train_on_expl:

        # if not fine-tuning on validation data, use the train_expl, valid_expl, test_expl
        # train_expl: Pre-computed explanation subgraphs for training triples and so on
        if not cfg.train.fine_tune_on_val:
            train_data.expl = dataset.train_expl
        valid_data.expl = dataset.valid_expl
        test_data.expl = dataset.test_expl

        # Normal case: train_data.target_edge_index contains all triples in the training set
        # but in explanation case, Replace with dataset.train_target_triples (a subset of triples that have explanations)
        if hasattr(dataset, "train_target_triples"):
            train_data.target_edge_index, train_data.target_edge_type = (
                dataset.train_target_triples[0],
                dataset.train_target_triples[1],
            )

        # Similarly, for validation and test data, replace with dataset.valid_target_triples and dataset.test_target_triples
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

    # if we are fine-tuning on validation data, we train on validation data, and fine-tune on validation data
    #$ Use 80% of validation for training, 20% for validation
    if cfg.train.fine_tune_on_val:
        # use 80% of the validation edges to finetune, and 20% to evaluate
        # if using DDP, this mask needs to be synchronized, s.t. the data is always the same
        if util.get_rank() == 0:
            train_mask = (
                torch.rand(valid_data.target_edge_index.size(1), device=device) > 0.2
            )
        else:
            train_mask = torch.zeros(
                valid_data.target_edge_index.size(1), device=device, dtype=torch.bool
            )
        if util.get_world_size() > 1:
            
            # share the mask across devices
            dist.all_reduce(train_mask, op=dist.ReduceOp.SUM)

        train_data = copy.copy(valid_data)
        train_data.target_edge_index = train_data.target_edge_index[:, train_mask]
        train_data.target_edge_type = train_data.target_edge_type[train_mask]
        valid_data.target_edge_index = valid_data.target_edge_index[:, ~train_mask]
        valid_data.target_edge_type = valid_data.target_edge_type[~train_mask]
        if util.get_rank() == 0:
            logger.warning(
                "Finetuning #train: %d, #valid: %d, #test: %d"
                % (
                    train_data.target_edge_index.size(1),
                    valid_data.target_edge_index.size(1),
                    test_data.target_edge_index.size(1),
                )
            )

    # * Random Walker Setup *
    # Random walk-based explanations (most cases: None)
    if model.rw_dropout:
        randomwalker = explainer_util.build_explainer(cfg, model, model.rw_dropout)
    else:
        randomwalker = None

    # * Filtering Data *
    if is_inductive:
        # for inductive setting, use only the test fact graph for filtered ranking
        filtered_data = None
    else:
        # for transductive setting, use the whole graph for filtered ranking
        filtered_data = Data(
            edge_index=dataset._data.target_edge_index,
            edge_type=dataset._data.target_edge_type,
            num_nodes=train_data.num_nodes,
        )
        filtered_data = filtered_data.to(device)

    # * Model Training *
    train_and_validate(
        cfg,
        model,
        train_data,
        valid_data,
        logger,
        device,
        filtered_data=filtered_data,
        randomwalker=randomwalker,
    )

    # * Evaluation *
    # disable wandb
    cfg.wandb.use = False

    # * Synethetic Dataset Evaluation *
    # Special evaluation: Only for synthetic datasets
    # WN18RR is not a synthetic dataset so it will not enter this block
    is_synthetic = cfg.dataset["class"] in ["synthetic", "aug_citation"]
    if is_synthetic:
        synthetic_evaluation(
            dataset, model, test_data, logger, device, filtered_data, working_dir
        )

    #$ Robustness Evaluation: Test model under different edge dropout conditions for valid and test data
    # * Evaluate Performance with Randomized Edge Drop*
    if cfg.eval.eval_on_edge_drop:
        for split_data, split_name in zip([valid_data, test_data], ["valid", "test"]):
            eval_on_edge_drop(
                model,
                split_data,
                split_name,
                logger,
                device,
                filtered_data,
                working_dir,
            )

    # * Evaluate Performance with Distrance Edge Drop *
    if cfg.eval.eval_on_distance_drop:
        for split_data, split_name in zip([valid_data, test_data], ["valid", "test"]):
            eval_on_distance_drop(
                cfg,
                model,
                split_data,
                split_name,
                logger,
                device,
                filtered_data,
                working_dir,
            )

    # * Standard Evaluation *
    # reset the randomized edge drop such that it evaluates on the full graph
    # skipped in config/nbfnet/wn18rr-0b.yaml
    #$ Turn off all special evaluation modes, Test on full, unmodified graph, Evaluate on train, valid, test (if enabled)

    model.eval_on_edge_drop = False
    model.distance_dropout = False
    model.randomized_edge_drop = 0
    model.eval_dropout_distance = -1
    if hasattr(cfg.train, "select_using_dist"):
        cfg.train.select_using_dist = False
    split_scores = {}
    for split_data, split_name in zip(
        [train_data, valid_data, test_data], ["train", "valid", "test"]
    ):
        if split_name == "train" and not cfg.eval.eval_on_train:
            continue
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning(f"Evaluate on {split_name}")
        _, scores = test(
            cfg,
            model,
            split_data,
            logger,
            device,
            filtered_data=filtered_data,
            working_dir=working_dir,
            split=split_name,
            final_evaluation=True,
        )
        split_scores[split_name] = scores

    # Standard evaluation results, return split_scores for aggregation across multiple runs
    if util.get_rank() == 0:
        df = pd.DataFrame(split_scores)
        df.to_csv(os.path.join(working_dir, f"evaluation.csv"))

    return split_scores


if __name__ == "__main__":


    # * Cfg, Dir, Logger Setup *
    # util ==> nbfnet/util.py where various util functions are defined

    # parse_args() ==> parses args in two phases: one for the args in the config file and the other for the args in the command line
    # args --> Controls how the script runs (seed, number of runs, etc.)
    # vars --> Controls what gets trained (model config, dataset paths, etc.)
    args, vars = util.parse_args()

    # load_config() ==> loads the config file into 'cfg'
    # cfg --> args + vars
    cfg = util.load_config(args.config, context=vars)

    # create_working_directory() ==> creates a directory for the experiment i.e. output_dir from config file
    # /scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG/experiments/NBFNet/WN18RR/2025-09-02-12-30-06
    working_dir = util.create_working_directory(cfg)

    # get_root_logger() ==> creates a logger for the experiment
    logger = util.get_root_logger()

    # avg_split_score --> stores the average scores for each split
    avg_split_score = {}


    # Get the value for X runs
    for i in range(args.num_runs):

        # change seed every run i.e. random initializations for each run
        args.seed += i

        #$ data loading, model setup, training, evaluation
        #$ returns: Dictionary of scores for each split
        '''
        {
        'train': {
            'mr': [45.2, 44.8, 46.1],      # Scores from run 0, 1, 2
            'mrr': [0.89, 0.91, 0.88],
            'hits@1': [0.82, 0.84, 0.81]
        },
        }
        '''
        split_scores = run(
            cfg, args, working_dir, logger, run_id=i, num_runs=args.num_runs
        )
        
        # Each run might create its own subdirectory so reset the working directory and logger
        if args.num_runs > 1:
            # set the dir back to the original in case we have a run-specific dir.
            os.chdir(working_dir)
            logger = util.change_logger_file(logger)
    
        # Main process will print the average scores
        if util.get_rank() == 0:
            for split_name, scores in split_scores.items():
                if split_name not in avg_split_score.keys():
                    avg_split_score[split_name] = defaultdict(list)
                for metric, score in scores.items():
                    avg_split_score[split_name][metric].append(score)

    # Afterwards, calculate the average and std.
    if util.get_rank() == 0:
        logger.warning(separator)
        all_split_scores = {}
        for split_name, avg_score in avg_split_score.items():
            logger.warning(f"{split_name} average scores")
            all_split_scores[split_name] = {}

            # avg_score --> { 'mr': [45.2, 44.8, 46.1], 'mrr': [0.89, 0.91, 0.88], 'hits@1': [0.82, 0.84, 0.81]}
            for metric, scores in avg_score.items():

                # calculate the average and std
                avg = np.average(scores)
                std = np.std(scores)

                logger.warning("avg_%s: %g, std: %g" % (metric, avg, std))
                all_split_scores[split_name][f"{metric}_avg"] = avg
                all_split_scores[split_name][f"{metric}_std"] = std

        # save the scores as a dataframe
        df = pd.DataFrame(all_split_scores)
        df.to_csv(os.path.join(working_dir, f"all_evaluation.csv"))
