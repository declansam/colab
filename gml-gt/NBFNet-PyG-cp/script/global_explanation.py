import os
import sys
import math
import pprint
import time
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


def evaluation(
    cfg,
    model,
    test_data,
    logger,
    device,
    filtered_data=None,
    working_dir=None,
    eval_split="valid",
    run=None,
    epoch=None,
):
    rank = util.get_rank()

    # * Logging Setup *
    eval_result = {}  # holds the average MRR for each eval type
    eval_stats = (
        {}
    )  # holds all the statistics for this evaluation, used to log best model metrics

    # * Hard Mask Evaluation *
    if "hard" in cfg.explainer_eval.eval_mask_type:
        mrr_stats = (
            {}
        )  # holds all the MRR info. In case of dual evaluation, this is used to calculate char. score.
        for eval_type in cfg.explainer_eval.eval_type:
            mrr_stats[eval_type] = {}
            with explainer_util.ExplainerEval(model, eval_type):
                ratios = getattr(cfg.explainer_eval, cfg.explainer_eval.eval_mask_type)
                results = []
                avg_score = defaultdict(list)
                for j, r in enumerate(ratios):
                    if rank == 0:
                        logger.warning(separator)
                        logger.warning(
                            f"Evaluate on {eval_split} with {eval_type}_{cfg.explainer_eval.eval_mask_type}: {r}"
                        )
                    result, scores, stats = test(
                        cfg,
                        model,
                        test_data,
                        logger,
                        device,
                        filtered_data=filtered_data,
                        working_dir=working_dir,
                        split=eval_split,
                        run=run,
                        epoch=epoch,
                        eval_type=eval_type,
                        eval_mask_type=cfg.explainer_eval.eval_mask_type,
                        ratio=r,
                        commit=False,
                    )
                    if rank == 0:
                        for metric, score in scores.items():
                            avg_score[metric].append(score)
                        eval_stats.update(
                            stats
                        )  # store the stat for this eval_type and ratio
                        mrr_stats[eval_type][
                            f"{cfg.explainer_eval.eval_mask_type}_{r}"
                        ] = result
                    results.append(result)
                result = sum(results) / len(results)  # the average MRR across ratios
                eval_result[eval_type] = result
                # * Wandb Log *
                if rank == 0:
                    logger.warning(separator)
                    stats = {}
                    # log the average score across validation sets
                    for metric, scores in avg_score.items():
                        avg_score = sum(scores) / len(scores)
                        stats[f"{eval_split}/avg_{metric}"] = avg_score
                        logger.warning("avg_%s: %g" % (metric, avg_score))
                    eval_stats.update(
                        stats
                    )  # store the average stats for this eval type
                    if cfg.wandb.use:
                        run.log(stats, step=epoch, commit=False)

    # * Soft Mask Evaluation *
    else:
        eval_result = {}
        for eval_type in cfg.explainer_eval.eval_type:
            with explainer_util.ExplainerEval(model, eval_type):
                result, _, stats = test(
                    cfg,
                    model,
                    test_data,
                    logger,
                    device,
                    filtered_data=filtered_data,
                    working_dir=working_dir,
                    split=eval_split,
                    run=run,
                    epoch=epoch,
                    eval_type=eval_type,
                    commit=False,
                )
                eval_stats.update(stats)
                eval_result[eval_type] = result

    # * Dual Evaluation *
    if len(cfg.explainer_eval.eval_type) == 2:
        # calculate the characterization score
        result = 1 / (
            (0.5 / eval_result["factual_eval"])
            + 0.5 / (1 - eval_result["counter_factual_eval"])
        )
        stats = {f"{eval_split}/avg_char_score": result.item()}

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
                    stats[f"{eval_split}/char_score_{ratio}"] = char_score.item()
            eval_stats.update(stats)  # store the characterization score
            # * Wandb Log *
            if cfg.wandb.use:
                run.log(stats, step=epoch, commit=False)

    return result, eval_stats


def train_and_validate(
    cfg,
    model,
    train_data,
    valid_data,
    logger,
    device,
    filtered_data=None,
    eval_split="valid",
):
    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()

    # * Wanbd Settings *
    run = util.wandb_setup(cfg, rank)

    # * Data Loading *
    train_triplets = torch.cat(
        [train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0)]
    ).t()
    # train_triplets = train_triplets[:16]
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
            result, eval_stats = evaluation(
                cfg,
                model,
                valid_data,
                logger,
                device,
                filtered_data=filtered_data,
                eval_split=eval_split,
                run=run,
                epoch=epoch,
            )

            # * Save Model *
            if (operator == "less_than" and result < best_result) or (
                operator == "more_than" and result > best_result
            ):
                best_result = result
                best_epoch = epoch
                if rank == 0:
                    logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
                    state = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    torch.save(state, "model_epoch_%d.pth" % epoch)
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
    state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device)
    model.load_state_dict(state["model"])
    util.synchronize()


@torch.no_grad()
def test_prediction(cfg, test_data, triples, mode, model, batch):
    """
    Get Explanation for Test Set, and Predict.
    Args:
        cfg: the config
        test_data: the test split data
        triples: the positive triples
        mode: the mode of evaluation
        model: the explainer model
        batch: the batch of triples to evaluate
    Returns:
        pred: the prediction for each triple
        node_mask: the mask of which node was included in the explanatory subgraph
        edge_mask: the mask of which edge was included in the explanatory subgraph
        num_edges: the number of edges in the explanatory subgraph
        num_nodes: the number of nodes in the explanatory subgraph
    """
    # get the topk predicted tails and the node and rel embeddings
    topk_tails, node_embeds, R_embeds = explainer_util.full_graph_prediction(
        test_data, triples, mode, model
    )
    heads = torch.where(mode, triples[:, 0], triples[:, 1])
    central_nodes = torch.cat((heads.unsqueeze(1), topk_tails), dim=-1)
    batch = tasks.conversion_to_tail_prediction(
        batch, model.eval_model.num_relation, mode
    )

    if model.ego_network:
        # prepare the ego data based on the central nodes
        data = data_util.prepare_ego_data(test_data, central_nodes, model.hops)
        batch = data_util.check_if_tail_in_network(data, batch)
        batch, valid_tails = data_util.relabel_batch(data, batch)
        # create batched_edge_index and edge_filter based on the masked data
        data = data_util.create_batched_data(data)
        node_embeds = data_util.create_s_node_embeds(data, node_embeds)
    else:
        data = data_util.prepare_full_data(test_data, len(batch), central_nodes)
        data = data_util.create_batched_data(data)  # create the batched data

    # do masking
    data = model.eval_model.masking(data, batch)

    if cfg.explainer["class"] == "PGExplainer":
        pred, node_mask, edge_mask, num_edges, num_nodes = model(
            data, batch, node_embeds
        )
    elif cfg.explainer["class"] == "RAWExplainer":
        pred, node_mask, edge_mask, num_edges, num_nodes = model(
            data, batch, node_embeds, R_embeds
        )

    if model.ego_network:
        pred[~valid_tails] = float(
            "-inf"
        )  # for the predictions outside the ego network, it will get a score of -inf
        # pred[~node_mask] = float('-inf') # for the predictions outside the explanatory subgraph, it will get a score of -inf
    return pred, node_mask, edge_mask, num_edges, num_nodes


@torch.no_grad()
def test(
    cfg,
    model,
    test_data,
    logger,
    device,
    filtered_data=None,
    working_dir=None,
    split="test",
    run=None,
    epoch=None,
    eval_type=None,
    eval_mask_type=None,
    ratio=None,
    commit=True,
):
    world_size = util.get_world_size()
    rank = util.get_rank()

    # * Data Loading *
    test_triplets = torch.cat(
        [test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]
    ).t()
    # test_triplets = test_triplets[:16]
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(
        test_triplets, cfg.train.batch_size, sampler=sampler
    )

    # * Model Setup *
    model.eval()
    if eval_mask_type is not None:
        setattr(model, eval_mask_type, ratio)

    # * Logging Setup *
    rankings = []
    inclusions = []
    num_negatives = []
    num_edges = []
    num_nodes = []
    modes = []  # logging the mode (1 for tail, 0 for head)
    heads = []
    rels = []  # logging the rel types.
    tails = []
    explanations = []
    if "save_explanation" in cfg and cfg.save_explanation:
        save_explanation = True
        assert world_size == 1
    else:
        save_explanation = False
        explanations = None

    # * Miscellaneous *
    is_synthetic = cfg.dataset["class"] in ["synthetic", "aug_citation"]
    if rank == 0:
        pbar = tqdm(total=len(test_loader))

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
        mode = torch.ones((batch_size,), dtype=torch.bool, device=batch.device)
        t_pred, t_node_mask, t_edge_mask, t_num_edges, t_num_nodes = test_prediction(
            cfg, test_data, triples=batch, mode=mode, model=model, batch=t_batch
        )
        # * Rank and Stat Computation *
        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        t_inclusion = t_node_mask[torch.arange(batch_size), pos_t_index]
        num_t_negative = t_mask.sum(dim=-1)

        # * Tail-Prediction only for Synthetic*
        if is_synthetic:
            # * Logging *
            rankings += [t_ranking]
            inclusions += [t_inclusion]
            num_negatives += [num_t_negative]
            num_edges += [t_num_edges]
            num_nodes += [t_num_nodes]
            heads += [batch[:, 0]]
            tails += [batch[:, 1]]
            rels += [batch[:, -1]]  # get the rels for both tail and head predictions
            modes += [
                torch.ones(t_ranking.shape)
            ]  # log whether it was for tail prediction or head prediction
            if save_explanation:
                explanations += [t_edge_mask.to("cpu")]
            if rank == 0:
                pbar.update(1)
            continue

        # * Head Prediction *
        mode = torch.zeros((batch_size,), dtype=torch.bool, device=batch.device)
        h_pred, h_node_mask, h_edge_mask, h_num_edges, h_num_nodes = test_prediction(
            cfg, test_data, triples=batch, mode=mode, model=model, batch=h_batch
        )
        # * Rank and Stat Computation *
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
        h_inclusion = h_node_mask[torch.arange(batch_size), pos_h_index]
        num_h_negative = h_mask.sum(dim=-1)

        # * Logging *
        rankings += [t_ranking, h_ranking]
        inclusions += [t_inclusion, h_inclusion]
        num_negatives += [num_t_negative, num_h_negative]
        num_edges += [t_num_edges, h_num_edges]
        num_nodes += [t_num_nodes, h_num_nodes]
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
        if save_explanation:
            explanations += [t_edge_mask.to("cpu"), h_edge_mask.to("cpu")]
        if rank == 0:
            pbar.update(1)
    if rank == 0:
        pbar.close()

    # * Aggr statistics across devices *
    stats = {
        "ranking": torch.cat(rankings),
        "inclusion": torch.cat(inclusions),
        "num_negative": torch.cat(num_negatives),
        "num_edges": torch.cat(num_edges),
        "num_nodes": torch.cat(num_nodes),
        "heads": torch.cat(heads),
        "tails": torch.cat(tails),
        "rels": torch.cat(rels),
        "modes": torch.cat(modes),
    }
    if save_explanation:
        explanations = torch.cat(explanations)

    all_stats, all_explanations = explainer_util.combine_stats(
        rank, world_size, device, stats
    )
    # * Calculate Metric *
    stats, scores = explainer_util.calc_metric_and_save_result(
        cfg,
        rank,
        eval_type,
        eval_mask_type,
        ratio,
        all_stats,
        split,
        run,
        epoch,
        commit,
        working_dir,
        save_explanation,
        all_explanations=all_explanations,
    )
    mrr = (1 / all_stats["ranking"].float()).mean()
    return mrr, scores, stats


def run(cfg, args):
    # * Dir, Logger Setup *
    working_dir = util.create_working_directory(cfg)
    torch.manual_seed(args.seed + util.get_rank())
    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
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

    if cfg.explainer["class"] == "PGExplainer" and explainer.mode == "transductive":
        train_data = valid_data

    train_and_validate(
        cfg,
        explainer,
        train_data,
        valid_data,
        logger,
        device,
        filtered_data=filtered_data,
    )

    if args.hyper_search:
        return  # do not get test performance if doing hypersearch
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
            eval_split="test",
        )
        return

    if "evaluate_on_train" in cfg and cfg.evaluate_on_train:
        splits = [[train_data, "train"], [valid_data, "valid"], [test_data, "test"]]
    else:
        splits = [[valid_data, "valid"], [test_data, "test"]]

    for data, split in splits:
        if util.get_rank() == 0:
            logger.warning(f"** Evaluation on {split} **")
        start_time = time.time()
        evaluation(
            cfg,
            explainer,
            data,
            logger,
            device,
            filtered_data=filtered_data,
            working_dir=working_dir,
            eval_split=split,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning(
                f"Execution time for inference on {split}: {elapsed_time:.1f} seconds"
            )


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    if args.hyper_search and not args.use_cfg:
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
