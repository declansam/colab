import os
import sys
import math
import pprint
import time
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import copy
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import tasks, util
from explainers import explainer_util, data_util
from explainers.BaseExplainer import CustomDDPWrapper

separator = ">" * 30
line = "-" * 30


def train_and_validate(
    cfg,
    model,
    train_data,
    valid_data,
    logger,
    device,
    run=None,
    filtered_data=None,
    working_dir=None,
    train_topk=None,
    valid_topk=None,
    eval_split="valid",
):
    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()

    # * Data Loading *
    train_triplets = torch.cat(
        [train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0)]
    ).t()
    # train_triplets = train_triplets[:30]
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
        parallel_model = CustomDDPWrapper(model, device_ids=[device])
    else:
        parallel_model = model

    # * Best Model Selection *
    if (
        len(cfg.explainer_eval.eval_mask_type) == 1
        and cfg.explainer_eval.eval_type[0] == "counter_factual_eval"
    ):
        best_result = float("inf")  # lower the metric, better it is for counter factual
        operator = "less_than"  # comparison operator
    else:
        best_result = float(
            "-inf"
        )  # higher the metric, better it is for factual / dual evaluation
        operator = "more_than"  # comparison operator
    best_epoch = -1

    # * Miscellaneous *
    if hasattr(cfg.train, "step"):
        step = cfg.train.step
    else:
        step = math.ceil(cfg.train.num_epoch / 10)
    is_synthetic = cfg.dataset["class"] in ["synthetic", "aug_citation"]
    batch_id = 0

    # * Training Loop *
    for epoch in range(0, cfg.train.num_epoch):
        # * Epoch Setup *
        parallel_model.set_train()
        sampler.set_epoch(epoch)
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning("Epoch %d begin" % epoch)
            pbar = tqdm(total=len(train_loader), desc="Processing Batch")

        # * Loss Dict *
        if cfg.explainer["class"] == "PGExplainer":
            loss_dict = {"bce_loss": [], "size_loss": [], "mask_ent_loss": []}
        elif cfg.explainer["class"] == "RAWExplainer":
            loss_dict = {
                "factual_loss": [],
                "counter_factual_loss": [],
                "size_loss": [],
                "mask_ent_loss": [],
                "rw_loss": [],
                "path_loss": [],
            }
        losses = []

        for batch in train_loader:
            # * Mode Setup *
            batch_size = len(batch)
            if is_synthetic:
                # for synthetic data, we do tail batch only
                mode = torch.ones((batch_size,), dtype=torch.bool, device=batch.device)
            else:
                mode = torch.zeros((batch_size,), dtype=torch.bool, device=batch.device)
                # first half will be tail batch, and the second half will be head batch
                mode[: batch_size // 2] = 1

            # * Explain *
            if cfg.explainer["class"] == "PGExplainer":
                loss = explainer_util.pgexplainer_train(
                    train_data, batch, mode, model, parallel_model, epoch
                )
                total_loss = 0
                for l, l_name in zip(loss, loss_dict.keys()):
                    loss_dict[l_name].append(l.item())
                    total_loss += l
            elif cfg.explainer["class"] == "RAWExplainer":
                loss = explainer_util.rawexplainer_train(
                    cfg,
                    train_data,
                    batch,
                    mode,
                    model,
                    parallel_model,
                    epoch,
                    is_synthetic,
                    train_topk=train_topk,
                )
                total_loss = 0
                for l, l_name in zip(loss, loss_dict.keys()):
                    loss_dict[l_name].append(l.item())
                    total_loss += l

            # * Backprop *
            loss = total_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # * Batch Log *
            if util.get_rank() == 0:
                pbar.update(1)
                if batch_id % cfg.train.log_interval == 0:
                    logger.warning(separator)
                    logger.warning("total loss: %g" % loss)
            losses.append(loss.item())
            batch_id += 1

        # * Epoch Log *
        if util.get_rank() == 0:
            avg_loss = sum(losses) / len(losses)
            logger.warning(separator)
            logger.warning("Epoch %d end" % epoch)
            logger.warning(line)
            logger.warning("average loss: %g" % avg_loss)
            pbar.close()

            # * Wandb Log *
            if cfg.wandb.use:
                if epoch % step == 0 or epoch == cfg.train.num_epoch - 1:
                    commit = False
                else:
                    commit = True
                stats = {}
                for l_name, l in loss_dict.items():
                    stats[f"train/{l_name}"] = sum(l) / len(l)
                stats["train/loss"] = avg_loss
                run.log(stats, step=epoch, commit=commit)

        util.synchronize()

        # * Model Validation *
        if epoch % step == 0 or epoch == cfg.train.num_epoch - 1:
            if cfg.explainer["class"] == "PGExplainer" and model.mode == "transductive":
                # When PGExplainer is used in the transductive setting, we only evaluate at the last epoch
                if epoch != cfg.train.num_epoch - 1:
                    if rank == 0 and cfg.wandb.use:
                        run.log({}, step=epoch, commit=True)
                    continue

            # * Validate Model *
            result, eval_stats = test(
                cfg,
                model,
                valid_data,
                logger,
                device,
                filtered_data=filtered_data,
                test_topk=valid_topk,
                working_dir=working_dir,
                split=eval_split,
                run=run,
                epoch=epoch,
            )

            # * Checkpoint *
            if not cfg.train.checkpoint_best and rank == 0:
                util.save_model(logger, epoch, model, optimizer)

            # * Log Best Result *
            if (operator == "less_than" and result < best_result) or (
                operator == "more_than" and result > best_result
            ):
                best_result = result
                best_epoch = epoch
                if rank == 0:
                    if cfg.train.checkpoint_best:
                        util.save_model(logger, epoch, model, optimizer)
                    # * Wandb Log *
                    if cfg.wandb.use:
                        stats = {}
                        for metric, score in eval_stats.items():
                            stats[f"best_{metric}"] = score
                        run.log(stats, step=epoch, commit=True)
            else:
                # * Empty Wandb Log *
                if rank == 0 and cfg.wandb.use:
                    run.log({}, step=epoch, commit=True)
    # * Load Best Model *
    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    util.synchronize()
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
    test_topk=None,
    working_dir=None,
    split="test",
    run=None,
    epoch=None,
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
    stats = explainer_util.logging_setup(cfg)
    if "save_explanation" in cfg and cfg.save_explanation and final_evaluation:
        # do not save explanation during model validation
        # we only test this for hard edge mask with budget so that we can pad it to the max edges.
        assert cfg.explainer_eval.eval_mask_type == "hard_edge_mask_top_k"
        save_explanation = True
    else:
        save_explanation = False
        explanations = None

    # * Evaluation Setup *
    eval_mask_type = cfg.explainer_eval.eval_mask_type
    ratios = getattr(cfg.explainer_eval, cfg.explainer_eval.eval_mask_type)

    # * Miscellaneous *
    is_synthetic = cfg.dataset["class"] in ["synthetic", "aug_citation"]
    if rank == 0:
        pbar = tqdm(total=len(test_loader))
    if (split == "train" and hasattr(cfg, "train_walltime")) or (
        split == "valid" and hasattr(cfg, "valid_walltime")
    ):
        walltime_seconds = (getattr(cfg, f"{split}_walltime") * 3600) / world_size
        if hasattr(cfg, "num_partitions"):
            walltime_seconds = walltime_seconds / cfg.num_partitions
    else:
        walltime_seconds = float("inf")
    start_time = time.time()
    processed_count = 1
    save_count = 0

    # * Test *
    for batch in test_loader:
        elapsed_time = time.time() - start_time
        if elapsed_time >= walltime_seconds:
            if rank == 0:
                logger.warning(f"Walltime reached.")
            explainer_util.periodic_saving(
                cfg,
                ratios,
                stats,
                save_explanation,
                rank,
                world_size,
                device,
                working_dir,
                save_count,
            )
            save_count += 1
            # reset the stats dictionary
            stats = explainer_util.logging_setup(cfg)
            util.synchronize()
            break

        # * Triple & Filtering Setup*
        batch_size = len(batch)
        t_batch, h_batch = tasks.all_negative(test_data, batch)
        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()

        # * Tail Prediction *
        mode = torch.ones((batch_size,), dtype=torch.bool, device=batch.device)

        # retain the original batch in case it's relabeld in the ego network process.
        query_batch = t_batch.clone()

        data, t_batch, node_embeds, R_embeds, valid_tails = (
            explainer_util.test_preparation(
                cfg,
                test_data,
                triples=batch,
                mode=mode,
                model=model,
                batch=t_batch,
                test_topk=test_topk,
            )
        )

        for eval_type in cfg.explainer_eval.eval_type:
            with explainer_util.ExplainerEval(model, eval_type):
                for r in ratios:
                    t_pred, t_node_mask, t_edge_mask, t_num_edges, t_num_nodes = (
                        explainer_util.test_prediction(
                            cfg,
                            data,
                            model,
                            batch=t_batch,
                            eval_mask_type=cfg.explainer_eval.eval_mask_type,
                            ratio=r,
                            node_embeds=node_embeds,
                            R_embeds=R_embeds,
                            valid_tails=valid_tails,
                        )
                    )
                    # assert torch.all(t_num_edges > 0)
                    if model.eval_mask_type == "hard_edge_mask_top_k":
                        assert torch.all(t_num_edges <= r)
                    # * Rank and Stat Computation *
                    ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
                    inclusion = t_node_mask[torch.arange(batch_size), pos_t_index]
                    num_negative = t_mask.sum(dim=-1)
                    stats[eval_type]["rankings"][r].append(ranking)
                    stats[eval_type]["inclusions"][r].append(inclusion)
                    stats[eval_type]["num_negatives"][r].append(num_negative)
                    stats[eval_type]["num_edges"][r].append(t_num_edges)
                    stats[eval_type]["num_nodes"][r].append(t_num_nodes)
                    stats[eval_type]["heads"][r].append(batch[:, 0])
                    stats[eval_type]["rels"][r].append(batch[:, -1])
                    stats[eval_type]["tails"][r].append(batch[:, 1])
                    stats[eval_type]["modes"][r].append(torch.ones(ranking.shape))
                    if save_explanation:
                        stats[eval_type]["explanations"][r].append(
                            explainer_util.get_expl_edge_indices(
                                query_batch, t_edge_mask, max_edges=r
                            )
                        )

        # * Tail-Prediction only for Synthetic*
        if is_synthetic:
            # * Logging *
            if rank == 0:
                pbar.update(1)
            continue

        # * Head Prediction *
        mode = torch.zeros((batch_size,), dtype=torch.bool, device=batch.device)

        # retain the original batch in case it's relabeld in the ego network process.
        query_batch = h_batch.clone()
        query_batch = tasks.conversion_to_tail_prediction(
            query_batch, model.eval_model.num_relation, mode
        )

        data, h_batch, node_embeds, R_embeds, valid_tails = (
            explainer_util.test_preparation(
                cfg,
                test_data,
                triples=batch,
                mode=mode,
                model=model,
                batch=h_batch,
                test_topk=test_topk,
            )
        )

        for eval_type in cfg.explainer_eval.eval_type:
            with explainer_util.ExplainerEval(model, eval_type):
                for r in ratios:
                    h_pred, h_node_mask, h_edge_mask, h_num_edges, h_num_nodes = (
                        explainer_util.test_prediction(
                            cfg,
                            data,
                            model,
                            batch=h_batch,
                            eval_mask_type=cfg.explainer_eval.eval_mask_type,
                            ratio=r,
                            node_embeds=node_embeds,
                            R_embeds=R_embeds,
                            valid_tails=valid_tails,
                        )
                    )
                    # assert torch.all(h_num_edges > 0)
                    if model.eval_mask_type == "hard_edge_mask_top_k":
                        assert torch.all(h_num_edges <= r)
                    # * Rank and Stat Computation *
                    ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
                    inclusion = h_node_mask[torch.arange(batch_size), pos_h_index]
                    num_negative = h_mask.sum(dim=-1)

                    stats[eval_type]["rankings"][r].append(ranking)
                    stats[eval_type]["inclusions"][r].append(inclusion)
                    stats[eval_type]["num_negatives"][r].append(num_negative)
                    stats[eval_type]["num_edges"][r].append(h_num_edges)
                    stats[eval_type]["num_nodes"][r].append(h_num_nodes)
                    stats[eval_type]["heads"][r].append(batch[:, 1])
                    stats[eval_type]["rels"][r].append(batch[:, -1])
                    stats[eval_type]["tails"][r].append(batch[:, 0])
                    stats[eval_type]["modes"][r].append(torch.zeros(ranking.shape))
                    if save_explanation:
                        stats[eval_type]["explanations"][r].append(
                            explainer_util.get_expl_edge_indices(
                                query_batch, h_edge_mask, max_edges=r
                            )
                        )

        if processed_count % (100 // world_size) == 0 or processed_count == len(
            test_loader
        ):
            explainer_util.periodic_saving(
                cfg,
                ratios,
                stats,
                save_explanation,
                rank,
                world_size,
                device,
                working_dir,
                save_count,
            )
            save_count += 1
            # reset the stats dictionary
            stats = explainer_util.logging_setup(cfg)
            util.synchronize()

        if rank == 0:
            pbar.update(1)
        processed_count += 1

    if rank == 0:
        pbar.close()

    # * Aggregate Statistics and Calculate Metric *
    # holds the average MRR for each eval type (Factual, Counterfactual)
    eval_result = {}
    # holds all the statistics for this evaluation, used to log best model metrics
    eval_stats = {}
    # holds all the MRR info. In case of dual evaluation, this is used to calculate char. score.
    mrr_stats = {}

    for eval_type in cfg.explainer_eval.eval_type:
        results = []
        mrr_stats[eval_type] = {}
        avg_score = defaultdict(list)
        for r in ratios:
            # make sure that all the result has been written to disk already.
            assert len(stats[eval_type]["rankings"][r]) == 0
            # * Aggr statistics across devices *
            if rank == 0:
                logger.warning(separator)

                # load the results from disk and combine.
                all_stats = defaultdict(list)
                all_explanations = []
                dir_path = os.path.join(working_dir, "saved_results")

                for i in range(save_count):
                    save_path = os.path.join(dir_path, f"result_{eval_type}_{r}_{i}.pt")
                    result = torch.load(save_path)
                    for key, var in result.items():
                        all_stats[key].append(var)

                    if save_explanation:
                        save_path = os.path.join(
                            dir_path, f"explanation_{eval_type}_{r}_{i}.pt"
                        )
                        explanation = torch.load(save_path)
                        all_explanations.append(explanation)

                for key, var in all_stats.items():
                    all_stats[key] = torch.cat(var)

                if save_explanation:
                    all_explanations = torch.cat(all_explanations)
                else:
                    all_explanations = None
                result = (1 / all_stats["ranking"].float()).mean()
                # * Calculate Metric *
                metric_stats, scores = explainer_util.calc_metric_and_save_result(
                    cfg,
                    rank,
                    eval_type,
                    eval_mask_type,
                    ratio=r,
                    all_stats=all_stats,
                    split=split,
                    run=run,
                    epoch=epoch,
                    commit=False,
                    working_dir=working_dir,
                    save_explanation=save_explanation,
                    all_explanations=all_explanations,
                    final_evaluation=final_evaluation,
                )
                # store the metric for this eval_type and ratio for logging best model
                eval_stats.update(metric_stats)
                # store each metric's score to calculate the average
                for metric, score in scores.items():
                    avg_score[metric].append(score)
            else:
                result = torch.tensor(0, device=device, dtype=torch.float32)

            if world_size > 1:
                # share the MRR result with other devices.
                dist.all_reduce(result, op=dist.ReduceOp.SUM)

            # * Log MRR *
            mrr_stats[eval_type][f"{cfg.explainer_eval.eval_mask_type}_{r}"] = result
            results.append(result)

        result = sum(results) / len(results)  # the average MRR across ratios
        eval_result[eval_type] = result

        if rank == 0:
            logger.warning(separator)
            avg_stats = {}
            for metric, scores in avg_score.items():
                avg_score = sum(scores) / len(scores)
                avg_stats[f"{split}/avg_{metric}"] = avg_score
                logger.warning("avg_%s: %g" % (metric, avg_score))
            # store the average stats across ratios for this eval type
            eval_stats.update(avg_stats)
            # * Wandb Log *
            if cfg.wandb.use:
                run.log(avg_stats, step=epoch, commit=False)

    # * Dual Evaluation *
    if len(cfg.explainer_eval.eval_type) == 2:
        # calculate the characterization score
        result = 1 / (
            (0.5 / eval_result["factual_eval"])
            + 0.5 / (1 - eval_result["counter_factual_eval"])
        )
        stats = {f"{split}/avg_char_score": result.item()}

        if rank == 0:
            logger.warning(separator)
            logger.warning("avg_char_score: %g" % (result.item()))
            # calculate the char score
            if "hard" in cfg.explainer_eval.eval_mask_type:
                for ratio, factual_score in mrr_stats["factual_eval"].items():
                    counter_factual_score = mrr_stats["counter_factual_eval"][ratio]
                    char_score = 1 / (
                        (0.5 / factual_score) + 0.5 / (1 - counter_factual_score)
                    )
                    logger.warning(f"char_score_{ratio}: %g" % (char_score.item()))
                    stats[f"{split}/char_score_{ratio}"] = char_score.item()
            eval_stats.update(stats)  # store the characterization score
            # * Wandb Log *
            if cfg.wandb.use:
                run.log(stats, step=epoch, commit=False)

    return result, eval_stats


def run(cfg, args):
    # * Dir, Logger Setup *
    dir = os.getcwd()
    working_dir = util.create_working_directory(cfg)
    torch.manual_seed(args.seed + util.get_rank())
    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(f"Process ID: {os.getpid()}")
        logger.warning(pprint.pformat(cfg))

    # * Dataset Setup *
    is_inductive = cfg.dataset["class"].startswith("Ind")
    dataset = util.build_dataset(cfg)

    # * Model, Explainer Setup *
    cfg.model.num_relation = dataset.num_relations
    model, eval_model = util.build_model_expl(cfg)
    explainer = explainer_util.build_explainer(cfg, model, eval_model)

    # * Send to Device *
    device = util.get_device(cfg)
    model = model.to(device)
    explainer = explainer.to(device)
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)

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

    if cfg.train.train_on_val or (
        cfg.explainer["class"] == "PGExplainer" and explainer.mode == "transductive"
    ):
        orig_train_data = train_data
        train_data = valid_data

    # * The TopK tails *
    if hasattr(cfg, "topk_dir"):
        topk_dict = {
            "train": dataset.train_topk,
            "valid": dataset.valid_topk,
            "test": dataset.test_topk,
        }
        if cfg.train.train_on_val:
            raise NotImplementedError
    else:
        topk_dict = {"train": None, "valid": None, "test": None}

    # * Wanbd Settings *
    wandb_run = util.wandb_setup(cfg, util.get_rank())

    train_and_validate(
        cfg,
        explainer,
        train_data,
        valid_data,
        logger,
        device,
        run=wandb_run,
        filtered_data=filtered_data,
        working_dir=working_dir,
        train_topk=topk_dict["train"],
        valid_topk=topk_dict["valid"],
    )

    # if args.hyper_search:
    #     return  # do not get test performance if doing hypersearch

    # disable wandb
    cfg.wandb.use = False

    if (
        cfg.explainer["class"] == "PGExplainer" and explainer.mode == "transductive"
    ):  # if PGExplainer and Transductive, we train and evaluate on test data
        train_and_validate(
            cfg,
            explainer,
            test_data,
            test_data,
            logger,
            device,
            filtered_data=filtered_data,
            working_dir=working_dir,
            eval_split="test",
        )
        return

    if "evaluate_on_train" in cfg and cfg.evaluate_on_train:
        if cfg.train.train_on_val:
            train_data = orig_train_data
        splits = [[train_data, "train"], [valid_data, "valid"], [test_data, "test"]]
    else:
        splits = [[valid_data, "valid"], [test_data, "test"]]

    for data, split in splits:
        if util.get_rank() == 0:
            logger.warning(f"** Evaluation on {split} **")
        start_time = time.time()
        test(
            cfg,
            explainer,
            data,
            logger,
            device,
            filtered_data=filtered_data,
            test_topk=topk_dict[split],
            working_dir=working_dir,
            split=split,
            final_evaluation=True,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning(
                f"Execution time for inference on {split}: {elapsed_time:.1f} seconds"
            )

    if args.train_config is not None:
        explainer_util.train_nbfnet(args, vars, dir, working_dir, logger, wandb_run)


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    if args.hyper_search:
        if (
            cfg.explainer["class"] == "PGExplainer"
            and cfg.explainer.mode == "transductive"
        ):
            cfg = torch.load("hyperparam_cfg_pgex_transductive.pt")
        elif (
            cfg.explainer["class"] == "PGExplainer"
            and cfg.explainer.mode == "inductive"
        ):
            cfg = torch.load("hyperparam_cfg_pgex_inductive.pt")
        elif cfg.explainer["class"] == "RAWExplainer":
            cfg = torch.load("hyperparam_cfg_rawexplainer.pt")
        run(cfg, args)
    else:
        run(cfg, args)
