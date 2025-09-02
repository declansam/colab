import os
import sys
import math
import pprint
from tqdm import tqdm
from collections import defaultdict
import copy
import time
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import tasks, util
from explainers import explainer_util
from explainers import data_util

separator = ">" * 30
line = "-" * 30


def train_explainer(split_data, batch, mode, model):
    """
    Trains Instance Explainer. GNNExplainer, PaGE-Link optimizes taking the top predicted tails
    as the answer, following original implementation.
    Args:
        split_data: the full split data
        batch: the true triples
        mode: the mode of evaluation
        model: the explainer model
    Returns:
        None
    """
    # set the model to training mode
    model.set_train()
    # get the topk predicted tails and the node and rel embeddings
    topk_tails, _, _ = explainer_util.full_graph_prediction(
        split_data, batch, mode, model
    )
    # create the central nodes (head & topk tails) for making the ego network
    heads = torch.where(mode, batch[:, 0], batch[:, 1])
    central_nodes = torch.cat((heads.unsqueeze(1), topk_tails), dim=-1)
    # prepare the triples for prediction, NBFNet will predict the score for the top tails
    pos_h_index, pos_t_index, pos_r_index = batch.t()
    h_index = pos_h_index.unsqueeze(-1).repeat(1, model.topk_tails)
    t_index = pos_t_index.unsqueeze(-1).repeat(1, model.topk_tails)
    r_index = pos_r_index.unsqueeze(-1).repeat(1, model.topk_tails)
    h_index = torch.where(
        mode.unsqueeze(1).repeat(1, model.topk_tails), h_index, topk_tails
    )
    t_index = torch.where(
        mode.unsqueeze(1).repeat(1, model.topk_tails), topk_tails, t_index
    )
    batch = torch.stack([h_index, t_index, r_index], dim=-1)
    # convert the batch to tail prediction
    batch = tasks.conversion_to_tail_prediction(
        batch, model.pred_model.num_relation, mode
    )

    if model.ego_network:  # explain from the ego network around the head and tail
        # prepare the ego data based on the central nodes
        data = data_util.prepare_ego_data(split_data, central_nodes, model.hops)
        # relabel the batch and also get the mask indicating the tails that are inside the ego network
        batch, valid_tails = data_util.relabel_batch(data, batch)
        heads = batch[:, 0, 0]
        tails = batch[:, :, 1]
        assert torch.equal(
            data.central_node_index, torch.cat((heads.unsqueeze(1), tails), dim=-1)
        ) and torch.all(valid_tails)
        # create batched_edge_index and edge_filter based on the masked data
        data = data_util.create_batched_data(data)
    else:
        data = data_util.prepare_full_data(split_data, len(batch), central_nodes)
        data = data_util.create_batched_data(data)  # create the batched data

    # do masking
    data = model.pred_model.masking(data, batch)
    model.train_mask(data, batch)


@torch.no_grad()
def test_prediction(
    cfg, test_data, triples, mode, model, batch, eval_mask_type=None, ratio=None
):
    """
    Get the prediction of NBFNet given the Explanatory subgraph. Assumes the explainer is already trained.
    Args:
        cfg: the config
        test_data: the full test data
        triples: the positive triples
        mode: the mode of evaluation
        model: the explainer model
        batch: the batch of triples to evaluate
        eval_mask_type: the evaluation mask type
        ratio: the budget for the evaluation mask type
    Returns:
        pred: the prediction for each triple
        node_mask: the mask of which node was included in the explanatory subgraph
        edge_mask: the mask of which edge was included in the explanatory subgraph
        num_edges: the number of edges in the explanatory subgraph
        num_nodes: the number of nodes in the explanatory subgraph
    """
    model.eval()
    if eval_mask_type is not None:
        setattr(model, eval_mask_type, ratio)
    # get the topk predicted tails and the node and rel embeddings
    topk_tails, _, _ = explainer_util.full_graph_prediction(
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
    else:
        data = data_util.prepare_full_data(test_data, len(batch), central_nodes)
        data = data_util.create_batched_data(data)  # create the batched data

    # do masking
    data = model.eval_model.masking(data, batch)
    pred, node_mask, edge_mask, num_edges, num_nodes = model.evaluate_mask(data, batch)

    if model.ego_network:
        # for the predictions outside the ego network, it will get a score of -inf
        pred[~valid_tails] = float("-inf")
        # pred[~node_mask] = float('-inf') # for the predictions outside the explanatory subgraph, it will get a score of -inf
    return pred, node_mask, edge_mask, num_edges, num_nodes


def set_budget(cfg, model, eval_mask_type, ratios):
    if eval_mask_type is not None and cfg.explainer["class"] == "PaGELink":
        # for PaGE-Link the process will retrieve paths walks after training
        # it will get the paths until it can satisfy the max quota or the time limit.
        setattr(model, eval_mask_type, max(ratios))


def reset_explainer(cfg, model):
    # resets the explainer for each instance
    if cfg.explainer["class"] == "RandomWalk":
        model.reset_node_dist()
        # setattr(model, "node_dist", None)


def explain(
    cfg,
    model,
    split_data,
    logger,
    device,
    filtered_data=None,
    working_dir=None,
    split="test",
    run=None,
    epoch=None,
    eval_mask_type=None,
    ratios=None,
):
    """
    Runs the explanation on instance-wise explainer.
    Args:
        cfg: the config data
        model: the explainer
        split_data: the split data
        logger: the logger
        device: the device for this process
        filtered_data: filtered data used for filtered ranking
        working_dir: the dir to save the result
        split: the name of the split
        run: wandb run object
        epoch: current epoch (should be 0 for instance explainer)
        eval_mask_type: the evaluation mask type
        ratios: the ratio used for hard masks.
    Returns:
        None
    """
    world_size = util.get_world_size()
    rank = util.get_rank()

    # * Data Loading *
    triplets = torch.cat(
        [split_data.target_edge_index, split_data.target_edge_type.unsqueeze(0)]
    ).t()
    # triplets = triplets[:32]
    sampler = torch_data.DistributedSampler(triplets, world_size, rank)
    loader = torch_data.DataLoader(triplets, cfg.train.batch_size, sampler=sampler)

    # * Logging Setup *
    stats = explainer_util.logging_setup(cfg)
    if "save_explanation" in cfg and cfg.save_explanation:
        save_explanation = True
        assert world_size == 1
    else:
        save_explanation = False
        explanations = None

    # * Miscellaneous *
    is_synthetic = cfg.dataset["class"] in ["synthetic", "aug_citation"]
    if rank == 0:
        pbar = tqdm(total=len(loader))

    # * Explain each instance *
    for batch in loader:
        # * Triple & Filtering Setup*
        batch_size = len(batch)
        t_batch, h_batch = tasks.all_negative(split_data, batch)
        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(split_data, batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()

        # * Tail Prediction *
        mode = torch.ones((batch_size,), dtype=torch.bool, device=batch.device)
        reset_explainer(cfg, model)  # reset the explainer for new instance
        set_budget(cfg, model, eval_mask_type, ratios)
        # train the explainer for these specific subgraphs & predictions
        train_explainer(split_data, batch, mode, model)

        for eval_type in cfg.explainer_eval.eval_type:
            with explainer_util.ExplainerEval(model, eval_type):
                for r in ratios:
                    t_pred, t_node_mask, t_edge_mask, t_num_edges, t_num_nodes = (
                        test_prediction(
                            cfg,
                            split_data,
                            triples=batch,
                            mode=mode,
                            model=model,
                            batch=t_batch,
                            eval_mask_type=eval_mask_type,
                            ratio=r,
                        )
                    )
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
                            t_edge_mask.to("cpu")
                        )

        if is_synthetic:
            if rank == 0:
                pbar.update(1)
            continue

        # * Head Prediction *
        mode = torch.zeros((batch_size,), dtype=torch.bool, device=batch.device)
        reset_explainer(cfg, model)  # reset the explainer for new instance
        set_budget(cfg, model, eval_mask_type, ratios)
        # train the explainer for these specific subgraphs & predictions
        train_explainer(split_data, batch, mode, model)
        for eval_type in cfg.explainer_eval.eval_type:
            with explainer_util.ExplainerEval(model, eval_type):
                for r in ratios:
                    h_pred, h_node_mask, h_edge_mask, h_num_edges, h_num_nodes = (
                        test_prediction(
                            cfg,
                            split_data,
                            triples=batch,
                            mode=mode,
                            model=model,
                            batch=h_batch,
                            eval_mask_type=eval_mask_type,
                            ratio=r,
                        )
                    )
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
                            h_edge_mask.to("cpu")
                        )

        if rank == 0:
            pbar.update(1)

    if rank == 0:
        pbar.close()

    # * Aggregate Statistics and Calculate Metric *
    # holds all the MRR info. In case of dual evaluation, this is used to calculate char. score.
    mrr_stats = {}
    for eval_type in cfg.explainer_eval.eval_type:
        mrr_stats[eval_type] = {}
        avg_score = defaultdict(list)
        for r in ratios:
            # * Aggr statistics across devices *
            if rank == 0:
                logger.warning(separator)
            statistics = {
                "ranking": torch.cat(stats[eval_type]["rankings"][r]),
                "inclusion": torch.cat(stats[eval_type]["inclusions"][r]),
                "num_negative": torch.cat(stats[eval_type]["num_negatives"][r]),
                "num_edges": torch.cat(stats[eval_type]["num_edges"][r]),
                "num_nodes": torch.cat(stats[eval_type]["num_nodes"][r]),
                "heads": torch.cat(stats[eval_type]["heads"][r]),
                "tails": torch.cat(stats[eval_type]["tails"][r]),
                "rels": torch.cat(stats[eval_type]["rels"][r]),
                "modes": torch.cat(stats[eval_type]["modes"][r]),
            }
            if save_explanation:
                explanations = torch.cat(stats[eval_type]["explanations"][r])

            all_stats = explainer_util.combine_stats(
                rank, world_size, device, statistics
            )
            # * Calculate Metric *
            _, scores = explainer_util.calc_metric_and_save_result(
                cfg,
                rank,
                eval_type,
                eval_mask_type,
                r,
                all_stats,
                split,
                run,
                epoch,
                False,
                working_dir,
                save_explanation,
                all_explanations=explanations,
            )
            for metric, score in scores.items():
                avg_score[metric].append(score)
            # * Log MRR *
            mrr = (1 / all_stats["ranking"].float()).mean()
            mrr_stats[eval_type][f"{cfg.explainer_eval.eval_mask_type}_{r}"] = mrr

        if rank == 0 and eval_mask_type is not None:
            logger.warning(separator)
            avg_stats = {}
            for metric, scores in avg_score.items():
                avg_score = sum(scores) / len(scores)
                avg_stats[f"{split}/avg_{metric}"] = avg_score
                logger.warning("average_%s: %g" % (metric, avg_score))
            # * Wandb Log *
            if cfg.wandb.use:
                run.log(avg_stats, step=epoch, commit=False)

    # * Dual Evaluation *
    if len(cfg.explainer_eval.eval_type) == 2 and rank == 0:
        char_stats = {}
        logger.warning(separator)
        # calculate the char score
        if "hard" in cfg.explainer_eval.eval_mask_type:
            factual_scores, counter_factual_scores = [], []
            for ratio, factual_score in mrr_stats["factual_eval"].items():
                counter_factual_score = mrr_stats["counter_factual_eval"][ratio]
                factual_scores.append(factual_score.item())
                counter_factual_scores.append(counter_factual_score.item())
                char_score = 1 / (
                    (0.5 / factual_score) + 0.5 / (1 - counter_factual_score)
                )
                logger.warning(f"char_score_{ratio}: %g" % (char_score.item()))
                char_stats[f"{split}/char_score_{ratio}"] = char_score.item()

        # calculate the average characterization score
        avg_factual_score = sum(factual_scores) / len(factual_scores)
        avg_counter_factual_score = sum(counter_factual_scores) / len(
            counter_factual_scores
        )
        result = 1 / (0.5 / avg_factual_score + 0.5 / (1 - avg_counter_factual_score))
        char_stats[f"{split}/avg_char_score"] = result
        logger.warning("avg_char_score: %g" % (result))

        # * Wandb Log *
        if cfg.wandb.use:
            run.log(char_stats, step=epoch, commit=True)

    return


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

    # * Wandb Setup *
    rank = util.get_rank()
    run = util.wandb_setup(cfg, rank)

    # * Explanation *
    if "hard" in cfg.explainer_eval.eval_mask_type:
        ratios = getattr(cfg.explainer_eval, cfg.explainer_eval.eval_mask_type)
        if rank == 0:
            logger.warning(separator)
            logger.warning(
                f"Evaluate on valid with {cfg.explainer_eval.eval_mask_type}"
            )
        start_time = time.time()
        explain(
            cfg,
            explainer,
            valid_data,
            logger,
            device,
            filtered_data=filtered_data,
            working_dir=working_dir,
            split="valid",
            run=run,
            epoch=0,
            eval_mask_type=cfg.explainer_eval.eval_mask_type,
            ratios=ratios,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning(
                f"Execution time for inference on valid: {elapsed_time:.1f} seconds"
            )
    else:
        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        explain(
            cfg,
            explainer,
            valid_data,
            logger,
            device,
            filtered_data=filtered_data,
            working_dir=working_dir,
            split="valid",
            run=run,
            epoch=0,
            ratios=[0],
        )

    if args.hyper_search:
        return  # do not get test performance if doing hypersearch
    # disable wandb
    cfg.wandb.use = False

    # * Test Set Explanation *
    if "hard" in cfg.explainer_eval.eval_mask_type:
        ratios = getattr(cfg.explainer_eval, cfg.explainer_eval.eval_mask_type)
        if rank == 0:
            logger.warning(separator)
            logger.warning(f"Evaluate on test with {cfg.explainer_eval.eval_mask_type}")
        explain(
            cfg,
            explainer,
            test_data,
            logger,
            device,
            filtered_data=filtered_data,
            working_dir=working_dir,
            split="test",
            run=run,
            epoch=0,
            eval_mask_type=cfg.explainer_eval.eval_mask_type,
            ratios=ratios,
        )
    else:
        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on test")
        explain(
            cfg,
            explainer,
            test_data,
            logger,
            device,
            filtered_data=filtered_data,
            working_dir=working_dir,
            split="test",
            run=run,
            epoch=0,
            ratios=[0],
        )


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    if args.hyper_search:
        # * Load random config for hypersearch *
        if cfg.explainer["class"] == "PaGELink":
            cfg = torch.load("hyperparam_cfg_pagelink.pt")
        elif cfg.explainer["class"] == "RandomWalk":
            cfg = torch.load("hyperparam_cfg_randomwalk.pt")
        elif cfg.explainer["class"] == "GNNExplainer":
            cfg = torch.load("hyperparam_cfg_gnnexplainer.pt")
        else:
            cfg = torch.load("hyperparam_cfg.pt")
        run(cfg, args)
    else:
        run(cfg, args)
