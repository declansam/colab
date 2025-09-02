import os
import sys
import math
import pprint
from tqdm import tqdm

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data
from torcheval.metrics import BinaryAUROC
from torcheval.metrics.toolkit import sync_and_compute

# Distributed evaluation: https://github.com/pytorch/torcheval/blob/main/examples/distributed_example.py

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import tasks, util
from hetero_rgcn.tasks import negative_sampling
from explainers import data_util, explainer_util


separator = ">" * 30
line = "-" * 30


def compute_metric(world_size, metric):
    if world_size > 1:
        auc = sync_and_compute(metric)
    else:
        auc = metric.compute()
    return auc


def rgcn_forward(split_data, batch, model, parallel_model=None, groundtruth_data=None):
    central_nodes = batch[:, :2].contiguous()
    label = batch[:, -1].to(torch.float64)
    data = data_util.prepare_ego_data_rgcn(split_data, central_nodes, model.hops)
    data = model.masking(data)
    if parallel_model is not None:
        pred = parallel_model(data, data.central_node_index.T)
    else:
        pred = model(data, data.central_node_index.T)
    return pred, label


def nbfnet_forward(
    split_data, batch, model, parallel_model=None, groundtruth_data=None
):
    central_nodes = batch[:, :2].contiguous()
    label = batch[:, -1].to(torch.float64)
    rels = batch[:, -2]
    data = data_util.prepare_ego_data(split_data, central_nodes, model.hops)
    data = data_util.map_groundtruth(data, groundtruth_data, central_nodes, label)
    data = data_util.create_batched_data(data)
    batch = torch.cat((data.central_node_index, rels.unsqueeze(1)), dim=-1).unsqueeze(1)
    data = model.masking(data, batch)
    if parallel_model is not None:
        pred = parallel_model(data, batch).squeeze()
    else:
        pred = model(data, batch).squeeze()
    return pred, label


def train_and_validate(
    cfg, model, train_data, valid_data, logger, device, groundtruth_data=None
):
    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()

    run = util.wandb_setup(cfg, rank)

    pos_edges = torch.cat(
        [
            train_data.target_edge_index,
            train_data.target_edge_type.unsqueeze(0),
            torch.ones(
                train_data.target_edge_index.size(1),
                device=train_data.target_edge_index.device,
                dtype=torch.int64,
            ).unsqueeze(0),
        ]
    ).T
    neg_edges = torch.cat(
        [
            train_data.neg_target_edge_index,
            train_data.neg_target_edge_type.unsqueeze(0),
            torch.zeros(
                train_data.neg_target_edge_index.size(1),
                device=train_data.neg_target_edge_index.device,
                dtype=torch.int64,
            ).unsqueeze(0),
        ]
    ).T

    if cfg.task.sample_neg_edges:
        train_edges = pos_edges
        # raise NotImplementedError
    else:
        train_edges = torch.cat([pos_edges, neg_edges], dim=0)

    sampler = torch_data.DistributedSampler(train_edges, world_size, rank)
    train_loader = torch_data.DataLoader(
        train_edges, cfg.train.batch_size, sampler=sampler
    )

    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)
    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    metric = BinaryAUROC(device=device)

    if hasattr(cfg.train, "step"):
        step = cfg.train.step
    else:
        step = math.ceil(cfg.train.num_epoch / 10)

    best_result = float("-inf")
    best_epoch = -1

    batch_id = 0
    for epoch in range(0, cfg.train.num_epoch):
        parallel_model.train()
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning("Epoch %d begin" % epoch)

        losses = []
        sampler.set_epoch(epoch)

        if rank == 0:
            pbar = tqdm(total=len(train_loader), desc="Processing Batch")

        for batch in train_loader:  # batch: (batch_size, 3)
            # create the ego network around the central nodes
            batch_size = batch.size(0)
            if cfg.model["class"] == "RGCN":
                pred, label = rgcn_forward(
                    train_data,
                    batch,
                    model,
                    parallel_model=parallel_model,
                    groundtruth_data=groundtruth_data,
                )
            elif cfg.model["class"] == "NBFNet":
                pred, label = nbfnet_forward(
                    train_data,
                    batch,
                    model,
                    parallel_model=parallel_model,
                    groundtruth_data=groundtruth_data,
                )

            if cfg.task.sample_neg_edges:
                # sample negative edges
                tmp = model.remove_ground_truth
                model.remove_ground_truth = (
                    False  # there are no ground truths for neg edges
                )
                batch = negative_sampling(train_data, batch_size)
                if cfg.model["class"] == "RGCN":
                    pred_neg, label_neg = rgcn_forward(
                        train_data, batch, model, parallel_model
                    )
                elif cfg.model["class"] == "NBFNet":
                    pred_neg, label_neg = nbfnet_forward(
                        train_data, batch, model, parallel_model
                    )
                model.remove_ground_truth = tmp  # put it back on

                pred = torch.cat((pred, pred_neg))
                label = torch.cat((label, label_neg))

            loss = F.binary_cross_entropy_with_logits(pred, label)
            metric.update(pred, label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if util.get_rank() == 0:
                pbar.update(1)

            if batch_id % cfg.train.log_interval == 0:
                auc = compute_metric(world_size, metric)
                if util.get_rank() == 0:
                    logger.warning(separator)
                    logger.warning("binary cross entropy: %g" % loss)
                    logger.warning(f"train AUROC: {auc}")
            losses.append(loss.item())
            batch_id += 1
            # if batch_id == 10:
            #     break

        auc = compute_metric(world_size, metric)
        if util.get_rank() == 0:
            avg_loss = sum(losses) / len(losses)
            logger.warning(separator)
            logger.warning("Epoch %d end" % epoch)
            logger.warning(line)
            logger.warning("average binary cross entropy: %g" % avg_loss)
            logger.warning(f"train AUROC: {auc}")
            pbar.close()
            if cfg.wandb.use:
                if epoch % step == 0 or epoch == cfg.train.num_epoch - 1:
                    commit = False
                else:
                    commit = True
                stats = {"train/loss": avg_loss, "train/auroc": auc}
                run.log(stats, step=epoch, commit=commit)

        metric.reset()

        # if rank == 0:
        #     logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
        #     state = {
        #         "model": model.state_dict(),
        #         "optimizer": optimizer.state_dict()
        #     }
        #     torch.save(state, "model_epoch_%d.pth" % epoch)
        util.synchronize()

        if epoch % step == 0 or epoch == cfg.train.num_epoch - 1:
            if rank == 0:
                logger.warning(separator)
                logger.warning("Evaluate on valid")
            result = test(
                cfg,
                model,
                valid_data,
                logger,
                device,
                split="valid",
                run=run,
                epoch=epoch,
                groundtruth_data=groundtruth_data,
            )
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
    groundtruth_data=None,
    working_dir=None,
    split="test",
    run=None,
    epoch=None,
    commit=True,
):
    world_size = util.get_world_size()
    rank = util.get_rank()

    pos_edges = torch.cat(
        [
            test_data.target_edge_index,
            test_data.target_edge_type.unsqueeze(0),
            torch.ones(
                test_data.target_edge_index.size(1),
                device=test_data.target_edge_index.device,
                dtype=torch.int64,
            ).unsqueeze(0),
        ]
    ).T
    neg_edges = torch.cat(
        [
            test_data.neg_target_edge_index,
            test_data.neg_target_edge_type.unsqueeze(0),
            torch.zeros(
                test_data.neg_target_edge_index.size(1),
                device=test_data.neg_target_edge_index.device,
                dtype=torch.int64,
            ).unsqueeze(0),
        ]
    ).T

    test_edges = torch.cat([pos_edges, neg_edges], dim=0)

    sampler = torch_data.DistributedSampler(test_edges, world_size, rank)
    test_loader = torch_data.DataLoader(
        test_edges, cfg.train.batch_size, sampler=sampler
    )

    metric = BinaryAUROC(device=device)

    if rank == 0:
        pbar = tqdm(total=len(test_loader))

    model.eval()

    for batch in test_loader:
        # create the ego network around the central nodes
        batch_size = batch.size(0)
        if cfg.model["class"] == "RGCN":
            pred, label = rgcn_forward(
                test_data, batch, model, groundtruth_data=groundtruth_data
            )
        elif cfg.model["class"] == "NBFNet":
            pred, label = nbfnet_forward(
                test_data, batch, model, groundtruth_data=groundtruth_data
            )

        metric.update(pred, label)

        if rank == 0:
            pbar.update(1)

    if rank == 0:
        pbar.close()

    auc = compute_metric(world_size, metric)
    if rank == 0:
        stats = {}
        for metric in cfg.task.metric:
            if metric == "auroc":
                score = auc
            stats[f"{split}/{metric}"] = score.item()
            logger.warning("%s: %g" % (metric, score))
        if cfg.wandb.use:
            run.log(stats, step=epoch, commit=commit)

    return auc


def run(cfg, args):
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    dataset = util.build_dataset(cfg)
    cfg.model.num_relation = dataset.num_relations
    if cfg.model["class"] == "RGCN":
        cfg.model.num_nodes = dataset[0].num_nodes
    model = util.build_model_expl(cfg)

    device = util.get_device(cfg)
    model = model.to(device)

    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    groundtruth_data = dataset.groundtruth_data
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)
    groundtruth_data = groundtruth_data.to(device)

    if args.randomized_edge_drop > 0:
        cfg.model.randomized_edge_drop = args.randomized_edge_drop
        model.randomized_edge_drop = args.randomized_edge_drop
        cfg.wandb.name += f"_{args.randomized_edge_drop}b"

    if model.remove_ground_truth:  # train with the groundtruth data removed
        train_and_validate(
            cfg,
            model,
            train_data,
            valid_data,
            logger,
            device,
            groundtruth_data=groundtruth_data,
        )
    else:
        train_and_validate(cfg, model, train_data, valid_data, logger, device)

    if args.hyper_search:
        return  # do not get test performance if doing hypersearch

    # disable wandb
    cfg.wandb.use = False

    # evaluate performance when ground truth is removed
    if cfg.eval.remove_ground_truth:
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
            groundtruth_data=groundtruth_data,
            working_dir=working_dir,
            split="test",
        )
        model.remove_ground_truth = False

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
                groundtruth_data=groundtruth_data,
                working_dir=working_dir,
                split="test",
            )
        # reset the randomized drop edge prob
        model.keep_ground_truth = False
        model.eval_on_edge_drop = False
        model.randomized_edge_drop = 0

    # evaluate performance with randomized edge drop
    if cfg.eval.eval_on_edge_drop:
        model.eval_on_edge_drop = True
        for randomized_edge_drop in cfg.eval.randomized_edge_drop:
            # for randomized_edge_drop in [1.0]:
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning(
                    f"Evaluate on valid with randomized edge drop {randomized_edge_drop}"
                )
            model.randomized_edge_drop = randomized_edge_drop
            test(
                cfg,
                model,
                valid_data,
                logger,
                device,
                working_dir=working_dir,
                split="valid",
            )
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning(
                    f"Evaluate on test with randomized edge drop {randomized_edge_drop}"
                )
            test(
                cfg,
                model,
                test_data,
                logger,
                device,
                working_dir=working_dir,
                split="test",
            )
        # reset the randomized drop edge prob
        model.eval_on_edge_drop = False
        model.randomized_edge_drop = 0

    # standard evaluation
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on valid")
    if model.remove_ground_truth:  # train with the groundtruth data removed
        test(
            cfg,
            model,
            valid_data,
            logger,
            device,
            working_dir=working_dir,
            split="valid",
            groundtruth_data=groundtruth_data,
        )
    else:
        test(
            cfg,
            model,
            valid_data,
            logger,
            device,
            working_dir=working_dir,
            split="valid",
        )
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")
    if model.remove_ground_truth:  # train with the groundtruth data removed
        test(
            cfg,
            model,
            test_data,
            logger,
            device,
            working_dir=working_dir,
            split="test",
            groundtruth_data=groundtruth_data,
        )
    else:
        test(
            cfg, model, test_data, logger, device, working_dir=working_dir, split="test"
        )


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    print("Hypersearch: ", args.hyper_search, flush=True)
    if args.hyper_search:
        cfg = torch.load("hyperparam_cfg_rgcn.pt")
        run(cfg, args)
    else:
        run(cfg, args)
