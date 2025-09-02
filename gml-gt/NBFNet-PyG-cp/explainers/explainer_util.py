import logging
import math
import copy
import os
import torch
import pandas as pd
from torch.nn import functional as F
from torch import distributed as dist
from collections import defaultdict

from explainers.GNNExplainer import GNNExplainer
from explainers.PaGELink import PaGELink
from explainers.PowerLink import PowerLink
from explainers.RandomWalk import RandomWalk
from explainers.PGExplainer import PGExplainer
from explainers.RAWExplainer import RAWExplainer

from explainers import data_util
from nbfnet import tasks, util
from script.train_nbfnet import run

logger = logging.getLogger(__file__)


def build_explainer(cfg, model, eval_model, rw_dropout=False):
    explainer_cfg = copy.deepcopy(cfg)
    cls = explainer_cfg.explainer.pop("class")

    if rw_dropout:
        assert cls == "RandomWalk"
        explainer = RandomWalk(None, **explainer_cfg.explainer)
    elif cls == "GNNExplainer":
        explainer = GNNExplainer(
            model,
            lr=cfg.optimizer.lr,
            optimizer=cfg.optimizer["class"],
            epochs=explainer_cfg.train.num_epoch,
            eval_mask_type=explainer_cfg.explainer_eval.eval_mask_type,
            model_to_evaluate=eval_model,
            **explainer_cfg.explainer,
        )
    elif cls == "PaGELink":
        explainer = PaGELink(
            model,
            lr=cfg.optimizer.lr,
            optimizer=cfg.optimizer["class"],
            epochs=explainer_cfg.train.num_epoch,
            eval_mask_type=explainer_cfg.explainer_eval.eval_mask_type,
            model_to_evaluate=eval_model,
            **explainer_cfg.explainer,
        )
    elif cls == "PowerLink":
        explainer = PowerLink(
            model,
            lr=cfg.optimizer.lr,
            optimizer=cfg.optimizer["class"],
            epochs=explainer_cfg.train.num_epoch,
            eval_mask_type=explainer_cfg.explainer_eval.eval_mask_type,
            model_to_evaluate=eval_model,
            **explainer_cfg.explainer,
        )
    elif cls == "RandomWalk":
        explainer = RandomWalk(
            model,
            eval_mask_type=explainer_cfg.explainer_eval.eval_mask_type,
            model_to_evaluate=eval_model,
            **explainer_cfg.explainer,
        )
    elif cls == "PGExplainer":
        explainer = PGExplainer(
            model,
            epochs=explainer_cfg.train.num_epoch,
            eval_mask_type=explainer_cfg.explainer_eval.eval_mask_type,
            model_to_evaluate=eval_model,
            **explainer_cfg.explainer,
        )
    elif cls == "RAWExplainer":
        explainer = RAWExplainer(
            model,
            epochs=explainer_cfg.train.num_epoch,
            eval_mask_type=explainer_cfg.explainer_eval.eval_mask_type,
            model_to_evaluate=eval_model,
            **explainer_cfg.explainer,
        )
    else:
        raise ValueError("Unknown Explainer `%s`" % cls)

    if "explainer_checkpoint" in explainer_cfg:
        state = torch.load(explainer_cfg.explainer_checkpoint, map_location="cpu")
        explainer.load_state_dict(state["model"])

    return explainer


def full_graph_prediction(full_data, batch, mode, model, topk=None):
    """
    Performs full graph prediction, get top k tails
    & the node and relation embeddings from the NBFNet
    Args:
        full_data: the full data
        batch: the true triples
        mode: the mode (True for tail and False for head)
        model: the explainer model (which holds the NBFNet)
        topk: the topk prediction by the GNN
    Returns:
        topk_tails: the topk predicted tails for each query
        embeds: the node embeddings for each query
        R: the relation embeddings
    """
    batch_size = len(batch)
    t_batch, h_batch = tasks.all_negative(full_data, batch)
    batch = torch.where(mode.view(batch_size, 1, 1), t_batch, h_batch)
    # prepare the ego data based on the heads
    data = data_util.prepare_full_data(full_data, batch_size)
    # convert the batch to tail prediction
    batch = tasks.conversion_to_tail_prediction(
        batch, model.pred_model.num_relation, mode
    )
    # do masking
    data = model.pred_model.masking(data, batch)
    # create batched_edge_index and edge_filter based on the masked data
    data = data_util.create_batched_data(data)

    if model.expl_gnn_model and topk is not None:
        # there is a GNN that gets the embeddings, skip full prediction
        # get the topk tails from the saved topk
        query = batch[:, 0, :]
        query = query[:, [0, 2]].to("cpu")
        all_query = topk[0]
        row_id, num_match = tasks.edge_match(all_query.T, query.T)
        assert torch.all(num_match == 1)
        topk_tails = topk[1][row_id][:, : model.topk_tails].to(batch.device)
        embeds, R = None, None

    else:
        original_pred, embeds, R = model.full_prediction(data, batch)
        tails = torch.argsort(original_pred, dim=1, descending=True)
        # option to filter out the other true tails, during training, you can also put the true tail
        topk_tails = tails[:, : model.topk_tails]
        # get the prediction of the topk tail
        """
        batch_id = torch.arange(batch_size).repeat_interleave(model.topk_tails)
        topk_tail_score = original_pred[batch_id, topk_tails.flatten()].sigmoid()
        """
    return topk_tails, embeds, R


def pgexplainer_train(train_data, batch, mode, model, parallel_model, epoch):
    """
    Training PGExplainer. Following the original implementation, PGExplainer will be optimized
    taking the top predicted tail as the answer.
    Args:
        train_data: the full training data
        batch: the true triples
        mode: the mode of the evaluation (True for tail, False for head)
        model: explainer model
        parallel_model: explainer model, could be DDP
        epoch: the epoch
    Returns:
        loss, size_loss, mask_ent_loss
    """
    # get the topk predicted tails and the node embeddings
    topk_tails, node_embeds, _ = full_graph_prediction(train_data, batch, mode, model)
    # create the central nodes for making the ego network
    heads = torch.where(mode, batch[:, 0], batch[:, 1])
    central_nodes = torch.cat((heads.unsqueeze(1), topk_tails), dim=-1)
    batch = batch.unsqueeze(1)
    # convert the batch to tail prediction
    batch = tasks.conversion_to_tail_prediction(
        batch, model.pred_model.num_relation, mode
    )
    # the NBFNet will predict the score for the top tails
    batch[:, :, 1] = topk_tails

    if model.ego_network:  # explain from the ego network around the head and tail
        # prepare the ego data based on the central nodes
        data = data_util.prepare_ego_data(train_data, central_nodes, model.hops)
        # relabel the batch and also get the mask indicating the tails that are inside the ego network
        batch, valid_tails = data_util.relabel_batch(data, batch)
        assert torch.equal(
            data.central_node_index, batch[:, :, :-1].squeeze(1)
        ) and torch.all(valid_tails)
        # create batched_edge_index and edge_filter based on the masked data
        data = data_util.create_batched_data(data)
        node_embeds = data_util.create_s_node_embeds(data, node_embeds)
    else:
        data = data_util.prepare_full_data(train_data, len(batch), central_nodes)
        data = data_util.create_batched_data(data)  # create the batched data

    # do masking
    data = model.pred_model.masking(data, batch)
    pred, size_loss, mask_ent_loss = parallel_model(data, batch, node_embeds, epoch)
    # compute the loss only with the valid tails
    if model.factual:  # factual explanation
        label = torch.ones_like(pred)
    else:
        label = torch.zeros_like(pred)  # counter factual
    loss = F.binary_cross_entropy_with_logits(pred, label, reduction="mean")
    return loss, size_loss, mask_ent_loss


def rawexplainer_train(
    cfg,
    train_data,
    batch,
    mode,
    model,
    parallel_model,
    epoch,
    is_synthetic=False,
    train_topk=None,
):
    """
    Training RAWExplainer for Factual Explanation. RAWExplainer will be optimized based on the true label.
    Args:
        cfg: the configuration
        train_data: the full training data
        batch: the true triples
        mode: the mode of the evaluation (True for tail, False for head)
        model: explainer model
        parallel_model: explainer model, could be DDP
        epoch: the epoch
        is_synthetic: whether the dataset is synthetic. Relevant for negative sampling mechanism.
        train_topk: the topk prediction by the GNN, used for RW.
    Returns:
        loss, size_loss, mask_ent_loss, rw_loss
    """
    # get the topk predicted tails and the node and rel embeddings
    topk_tails, node_embeds, R_embeds = full_graph_prediction(
        train_data, batch, mode, model, topk=train_topk
    )
    # create the central nodes for making the ego network
    heads = torch.where(mode, batch[:, 0], batch[:, 1])
    central_nodes = torch.cat((heads.unsqueeze(1), topk_tails), dim=-1)
    if model.ego_network:  # explain from the ego network around the head and tail
        # prepare the ego data based on the central nodes
        data = data_util.prepare_ego_data(train_data, central_nodes, model.hops)
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
            batch, model.pred_model.num_relation, mode
        )
        # relabel the batch and also get the mask indicating the tails that are inside the ego network
        batch, valid_tails = data_util.relabel_batch(data, batch)
        data = data_util.create_batched_data(data)
        node_embeds = data_util.create_s_node_embeds(data, node_embeds)
    else:
        batch = tasks.negative_sampling(
            train_data,
            batch,
            cfg.task.num_negative,
            strict=cfg.task.strict_negative,
            is_synthetic=is_synthetic,
            mode=mode,
        )
        # convert the batch to tail prediction
        batch = tasks.conversion_to_tail_prediction(
            batch, model.pred_model.num_relation, mode
        )
        data = data_util.prepare_full_data(train_data, len(batch), central_nodes)
        data = data_util.create_batched_data(data)  # create the batched data

    # do masking
    data = model.pred_model.masking(data, batch)
    # create batched_edge_index and edge_filter based on the masked datas
    preds, size_loss, mask_ent_loss, rw_loss, path_loss = parallel_model(
        data, batch, node_embeds, R_embeds, epoch
    )

    factual_loss, counter_factual_loss = torch.zeros_like(size_loss), torch.zeros_like(
        size_loss
    )
    if model.ego_network:
        if model.factual:
            pred = preds["factual"]
            target = torch.zeros_like(pred, dtype=torch.bool)
            target[:, 0] = 1
            factual_loss = compute_loss_ego(
                cfg, pred, target, pos_included, valid_tails
            )

        if model.counter_factual:
            pred = preds["counter_factual"]
            target = torch.zeros_like(pred, dtype=torch.bool)
            counter_factual_loss = compute_loss_ego(
                cfg, pred, target, pos_included, valid_tails
            )

    else:
        if model.factual:
            pred = preds["factual"]
            target = torch.zeros_like(pred)
            target[:, 0] = 1
            factual_loss = compute_loss(cfg, pred, target)

        if model.counter_factual:
            pred = preds["counter_factual"]
            target = torch.zeros_like(pred)
            counter_factual_loss = compute_loss(cfg, pred, target)

    return (
        factual_loss,
        counter_factual_loss,
        size_loss,
        mask_ent_loss,
        rw_loss,
        path_loss,
    )


def compute_loss(cfg, pred, target):
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    neg_weight = torch.ones_like(pred)
    if cfg.task.adversarial_temperature > 0:
        with torch.no_grad():  # the higher the score is for a negative pred, the more it will contribute
            neg_weight[:, 1:] = F.softmax(
                pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1
            )
    else:
        neg_weight[:, 1:] = 1 / cfg.task.num_negative
    loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
    loss = loss.mean()
    return loss


def compute_loss_ego(cfg, pred, is_pos, pos_included, valid_tails):
    # sometimes the pos tail is not included in the ego network
    is_pos[~pos_included] = 0
    is_neg = ~is_pos
    # sometimes there is no neg tail to sample from the ego_network (singleton)
    is_neg[~valid_tails] = 0
    # check there are no mismatchs
    assert torch.all(
        is_pos.to(torch.long) + is_neg.to(torch.long) + (~valid_tails).to(torch.long)
        == 1
    )
    # compute the loss only with the valid tails
    loss = F.binary_cross_entropy_with_logits(
        pred, is_pos.to(pred.dtype), reduction="none"
    )
    if cfg.task.adversarial_temperature > 0:
        with torch.no_grad():  # the higher the score is for a negative pred, the more it will contribute
            neg_weight = pred / cfg.task.adversarial_temperature
            # pos and invalid tails will get -inf to not affect the softmax
            neg_weight[~is_neg] = float("-inf")
            neg_weight = F.softmax(neg_weight, dim=-1)
            neg_weight[is_pos] = 1  # pos will get 1
    else:
        num_negative = is_neg.sum(dim=-1)
        neg_weight = torch.ones_like(pred)
        neg_weight = neg_weight / num_negative.unsqueeze(1)
        neg_weight[is_pos] = 1

    # the invalid tails will get a score of 0 to nullify the loss contribution
    neg_weight[~valid_tails] = 0
    loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
    loss = loss.mean()
    return loss


@torch.no_grad()
def test_preparation(
    cfg, test_data, triples, mode, model, batch, test_topk=None, data=None
):
    """
    This function does the static component of testing that doesn't change across
    evaluation of different ratios for the same batch.
    Namely, this function is responsible for:
        1. full graph prediction and embedding retrieval
        2. data processing by creating batched data
        3. Any masking if necessary
    Args:
        cfg: the config
        test_data: the test split data
        triples: the positive triples
        mode: the mode of evaluation
        model: the explainer model
        batch: the batch of triples to evaluate
        test_topk: the topk predicted entities
        data: for instance explainers, the mask is trained already on a fixed subgraph.
                use the same subgraph to not have issues.
    Returns:
        data: the processed data
        batch: the processed batch
        node_embeds: the node embedding obtained by the original model
        R_embeds: the relation embedding obtained by the original model
        valid_tails: the tails that are not inside the ego-network. If not
        using ego_network, this returns None.
    """
    if data is not None:
        batch = tasks.conversion_to_tail_prediction(
            batch, model.eval_model.num_relation, mode
        )
        if model.ego_network:
            # prepare the ego data based on the central nodes
            batch = data_util.check_if_tail_in_network(data, batch)
            batch, valid_tails = data_util.relabel_batch(data, batch)
        else:
            valid_tails = None

        return data.detach(), batch, None, None, valid_tails

    # get the topk predicted tails and the node and rel embeddings
    topk_tails, node_embeds, R_embeds = full_graph_prediction(
        test_data, triples, mode, model, topk=test_topk
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
        valid_tails = None

    # do masking
    data = model.eval_model.masking(data, batch)

    return data, batch, node_embeds, R_embeds, valid_tails


@torch.no_grad()
def test_prediction(
    cfg,
    data,
    model,
    batch,
    eval_mask_type=None,
    ratio=None,
    node_embeds=None,
    R_embeds=None,
    valid_tails=None,
):
    """
    Get Explanation for Test Set, and Predict.
    Args:
        cfg: the config
        data: the processed data
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
    # deep copy the data s.t. the processed data remains the same
    data = copy.deepcopy(data)
    # * Model Setup *
    model.eval()
    if eval_mask_type is not None:
        setattr(model, eval_mask_type, ratio)

    if cfg.explainer["class"] == "PGExplainer":
        pred, node_mask, edge_mask, num_edges, num_nodes = model.evaluate_mask(
            data, batch, node_embeds
        )
    elif cfg.explainer["class"] == "RAWExplainer":
        pred, node_mask, edge_mask, num_edges, num_nodes = model.evaluate_mask(
            data, batch, node_embeds, R_embeds
        )
    elif cfg.explainer["class"] in [
        "GNNExplainer",
        "PaGELink",
        "RandomWalk",
        "PowerLink",
    ]:
        pred, node_mask, edge_mask, num_edges, num_nodes = model.evaluate_mask(
            data, batch
        )
    else:
        raise ValueError

    if model.ego_network:
        # for the predictions outside the ego network, it will get a score of -inf
        pred[~valid_tails] = float("-inf")
        # pred[~node_mask] = float('-inf') # for the predictions outside the explanatory subgraph, it will get a score of -inf
    return pred, node_mask, edge_mask, num_edges, num_nodes


def combine_stats(rank, world_size, device, stats, explanations=None):
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(stats["ranking"])
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
    cum_size = all_size.cumsum(0)

    all_stats = {}
    for name, stat in stats.items():
        all_stat = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
        all_stat[cum_size[rank] - all_size[rank] : cum_size[rank]] = stat
        if world_size > 1:
            dist.all_reduce(all_stat, op=dist.ReduceOp.SUM)
        all_stats[name] = all_stat

    if explanations is not None:
        all_explanations = torch.zeros(
            all_size.sum(),
            explanations.size(1),
            dtype=explanations.dtype,
            device=device,
        )
        all_explanations[cum_size[rank] - all_size[rank] : cum_size[rank], :] = (
            explanations
        )
        if world_size > 1:
            dist.all_reduce(all_explanations, op=dist.ReduceOp.SUM)
    else:
        all_explanations = None

    return all_stats, all_explanations


def calc_metric_and_save_result(
    cfg,
    rank,
    eval_type,
    eval_mask_type,
    ratio,
    all_stats,
    split,
    run=None,
    epoch=None,
    commit=False,
    working_dir=None,
    save_explanation=False,
    all_explanations=None,
    save_topk=False,
    topk=None,
    final_evaluation=False,
    distance=None,
):
    stats = {}
    scores = {}
    if rank == 0:
        eval_type_info = ""
        additional_info = ""
        if eval_type is not None:
            eval_type_info += f"_{eval_type}"
            additional_info += f"_{eval_type}"
        if eval_mask_type is not None:
            additional_info += f"_{eval_mask_type}_{ratio}"
        if (
            hasattr(cfg.train, "select_using_dist")
            and cfg.train.select_using_dist
            and distance is not None
        ):
            additional_info += f"_{distance}_dist"
        for metric in cfg.task.metric:
            if metric == "mr":
                score = all_stats["ranking"].float().mean()
            elif metric == "mrr":
                score = (1 / all_stats["ranking"].float()).mean()
            elif metric.startswith("hits@"):
                values = metric[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    # unbiased estimation
                    fp_rate = (all_stats["ranking"] - 1).float() / all_stats[
                        "num_negative"
                    ]
                    score = 0
                    for i in range(threshold):
                        # choose i false positive from num_sample - 1 negatives
                        num_comb = (
                            math.factorial(num_sample - 1)
                            / math.factorial(i)
                            / math.factorial(num_sample - i - 1)
                        )
                        score += (
                            num_comb
                            * (fp_rate**i)
                            * ((1 - fp_rate) ** (num_sample - i - 1))
                        )
                    score = score.mean()
                else:
                    score = (all_stats["ranking"] <= threshold).float().mean()
            elif metric in ["inclusion", "num_edges", "num_nodes"]:
                score = all_stats[metric].float().mean()
            stats[f"{split}/{metric}{additional_info}"] = score.item()
            scores[metric + eval_type_info] = score.item()
            logger.warning("%s: %g" % (metric + additional_info, score))
        # * Wandb Log *
        if cfg.wandb.use:
            run.log(stats, step=epoch, commit=commit)
        # * Save Result *
        if working_dir is not None and final_evaluation:
            data = {
                "Ranking": all_stats["ranking"].tolist(),
                "Heads": all_stats["heads"].tolist(),
                "Tails": all_stats["tails"].tolist(),
                "Rel": all_stats["rels"].tolist(),
                "Mode": all_stats["modes"].tolist(),
            }
            try:
                expl_data = {
                    "Num_Edges": all_stats["num_edges"].tolist(),
                    "Num_Nodes": all_stats["num_nodes"].tolist(),
                    "Inclusion": all_stats["inclusion"].tolist(),
                }
                data.update(expl_data)
            except KeyError:
                pass
            torch.save(
                data, os.path.join(working_dir, f"{split}_output{additional_info}.pt")
            )
            if save_explanation:
                torch.save(
                    all_explanations,
                    os.path.join(
                        working_dir, f"{split}_explanations{additional_info}.pt"
                    ),
                )
            if save_topk:
                torch.save(
                    topk, os.path.join(working_dir, f"{split}_topk{additional_info}.pt")
                )

    return stats, scores


class ExplainerEval:
    def __init__(self, explainer, attr_name):
        self.explainer = explainer
        self.original_value = getattr(explainer, attr_name)
        assert self.original_value == False
        self.attr_name = attr_name

    def __enter__(self):
        setattr(self.explainer, self.attr_name, True)

    def __exit__(self, type, value, traceback):
        setattr(self.explainer, self.attr_name, self.original_value)


def logging_setup(cfg, explanations=True):
    stats = {}
    for eval_type in cfg.explainer_eval.eval_type:
        stats[eval_type] = {
            "rankings": defaultdict(list),
            "inclusions": defaultdict(list),
            "num_negatives": defaultdict(list),
            "num_edges": defaultdict(list),
            "num_nodes": defaultdict(list),
            "modes": defaultdict(list),  # logging the mode (1 for tail, 0 for head)
            "heads": defaultdict(list),
            "rels": defaultdict(list),  # logging the rel types.
            "tails": defaultdict(list),
        }
        if explanations:
            stats[eval_type]["explanations"] = defaultdict(list)
    return stats


def get_expl_edge_indices(batch, edge_mask, max_edges):
    """
    Saves the explanation edge indices.
    The first two columns will be (head, rel).
    The next columns will be the edge_id for the chosen edges.
    The tensors will be of size max_edges. -1 will be used for padding
    """
    h_index, t_index, r_index = batch.unbind(-1)  # (batch_size, num_triples)
    # assumption: the batch has been converted to tail batch
    assert (h_index[:, [0]] == h_index).all()
    assert (r_index[:, [0]] == r_index).all()
    # prepare the query (h, r)
    query = batch[:, 0, :]
    query = query[:, [0, 2]].to(torch.int32)
    # create edge indices tensor that has everything padded with -1
    edge_indices = torch.zeros(
        (query.size(0), max_edges), dtype=torch.int32, device=query.device
    ).fill_(-1)
    # get the number of edges
    num_edges = edge_mask.sum(1)
    indices = torch.arange(max_edges, device=query.device).repeat(query.size(0), 1)
    edge_filter = indices < num_edges.unsqueeze(1)
    # fill in the selected edges
    edge_indices[edge_filter] = edge_mask.nonzero()[:, 1].to(torch.int32)
    edge_indices = torch.cat((query, edge_indices), dim=1)
    return edge_indices


separator = ">" * 30


def train_nbfnet(args, vars, dir, working_dir, logger, wandb_run):
    # * Cfg, Dir, Logger Setup *
    cfg = util.load_config(os.path.join(dir, args.train_config), context=vars)
    # disable wandb for the fine-tuning
    cfg.wandb.use = False
    # pass the current dir as the saved expl dir
    cfg.explainer_eval.expl_dir = working_dir
    ratios = getattr(cfg.explainer_eval, cfg.explainer_eval.eval_mask_type)
    if not isinstance(ratios, list):
        ratios = [ratios]

    avg_score = {}
    for ratio in ratios:
        setattr(cfg.explainer_eval, cfg.explainer_eval.eval_mask_type, ratio)
        additional_info = f"{cfg.explainer_eval.eval_mask_type}_{ratio}"
        split_scores = run(
            cfg,
            args,
            working_dir,
            logger,
            run_id=additional_info,
            log_separate=True,
        )
        # reset the working dir and logger
        os.chdir(working_dir)
        logger = util.change_logger_file(logger)
        if util.get_rank() == 0:
            stats = {}
            for split, scores in split_scores.items():
                if split not in avg_score.keys():
                    avg_score[split] = defaultdict(list)
                for metric, score in scores.items():
                    stats[
                        f"{cfg.model['class']}_{split}/{metric}_{additional_info}"
                    ] = score
                    avg_score[split][metric].append(score)
            if wandb_run is not None:
                wandb_run.summary.update(stats)
    if util.get_rank() == 0:
        logger.warning(separator)
        stats = {}
        for split, scores in avg_score.items():
            for metric, score in scores.items():
                stats[f"{cfg.model['class']}_{split}/avg_{metric}"] = sum(score) / len(
                    score
                )
                logger.warning(
                    f"{cfg.model['class']}_{split}/avg_{metric}: {sum(score)/len(score)}"
                )
            df = pd.DataFrame(scores, index=ratios)
            df.to_csv(os.path.join(working_dir, f"{split}_all_evaluation.csv"))

        if wandb_run is not None:
            wandb_run.summary.update(stats)


def periodic_saving(
    cfg,
    ratios,
    stats,
    save_explanation,
    rank,
    world_size,
    device,
    working_dir,
    save_count,
):
    # periodically combine the statistics (if DDP) and write it to disk.
    for eval_type in cfg.explainer_eval.eval_type:
        for r in ratios:
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
            else:
                explanations = None
            all_stats, all_explanations = combine_stats(
                rank, world_size, device, statistics, explanations
            )
            # write it to disk
            if rank == 0:
                dir_path = os.path.join(working_dir, "saved_results")
                if not os.path.isdir(dir_path):
                    os.makedirs(dir_path)
                save_path = os.path.join(
                    dir_path, f"result_{eval_type}_{r}_{save_count}.pt"
                )
                torch.save(all_stats, save_path)
                if save_explanation:
                    save_path = os.path.join(
                        dir_path, f"explanation_{eval_type}_{r}_{save_count}.pt"
                    )
                    torch.save(all_explanations, save_path)
