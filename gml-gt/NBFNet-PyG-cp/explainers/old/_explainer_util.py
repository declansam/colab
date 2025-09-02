import logging
import math
import copy
import os
import torch
from torch_geometric.loader import DataLoader
from torch.nn import functional as F
from torch import distributed as dist

from explainers.GNNExplainer import GNNExplainer
from explainers.PaGELink import PaGELink
from explainers.RandomWalk import RandomWalk
from explainers.PGExplainer import PGExplainer
from explainers.RAWExplainer import RelPGExplainer

# from explainers.DynamicRelPGExplainer import DynamicRelPGExplainer
from explainers.DynamicEgoRelPGExplainer import DynamicEgoRelPGExplainer
from explainers._explanation_dataset import (
    ExplanationDataset,
    DynamicExplanationDataset,
)
from explainers import data_util
from nbfnet import tasks

logger = logging.getLogger(__file__)


def index_to_mask(index, size):
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask


def build_explainer(cfg, model, rw_dropout=False):
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
            **explainer_cfg.explainer,
        )
    elif cls == "PaGELink":
        explainer = PaGELink(
            model,
            lr=cfg.optimizer.lr,
            optimizer=cfg.optimizer["class"],
            epochs=explainer_cfg.train.num_epoch,
            eval_mask_type=explainer_cfg.explainer_eval.eval_mask_type,
            **explainer_cfg.explainer,
        )
    elif cls == "RandomWalk":
        explainer = RandomWalk(
            model,
            eval_mask_type=explainer_cfg.explainer_eval.eval_mask_type,
            **explainer_cfg.explainer,
        )
    elif cls == "PGExplainer":
        explainer = PGExplainer(
            model,
            epochs=explainer_cfg.train.num_epoch,
            eval_mask_type=explainer_cfg.explainer_eval.eval_mask_type,
            **explainer_cfg.explainer,
        )
    elif cls == "RelPGExplainer":
        explainer = RelPGExplainer(
            model,
            epochs=explainer_cfg.train.num_epoch,
            eval_mask_type=explainer_cfg.explainer_eval.eval_mask_type,
            **explainer_cfg.explainer,
        )
    else:
        raise ValueError("Unknown Explainer `%s`" % cls)

    if "explainer_checkpoint" in explainer_cfg:
        state = torch.load(explainer_cfg.explainer_checkpoint, map_location="cpu")
        explainer.load_state_dict(state["model"])

    return explainer


def build_ego_explainer(cfg, model):
    cls = cfg.explainer.pop("class")

    if hasattr(cfg.explainer_eval, "eval_on_random"):
        eval_on_random = cfg.explainer_eval.eval_on_random
    else:
        eval_on_random = False

    if cls == "RelPGExplainer":
        explainer = DynamicEgoRelPGExplainer(
            model,
            epochs=cfg.train.num_epoch,
            eval_mask_type=cfg.explainer_eval.eval_mask_type,
            eval_on_random=eval_on_random,
            **cfg.explainer,
        )
    else:
        raise ValueError("Unknown Explainer `%s`" % cls)

    if "explainer_checkpoint" in cfg:
        state = torch.load(cfg.explainer_checkpoint, map_location="cpu")
        explainer.load_state_dict(state["model"])

    return explainer


def build_explainer_dataset(cfg, dataset, evaluation, hops):
    cls = cfg.dataset["class"]
    root = cfg.dataset.root

    explanation_datasets = []
    if evaluation == "inference-test-only":
        expl_dataset = ExplanationDataset(
            root=root, name=cls, dataset=dataset, split="test", hops=hops
        )
        explanation_datasets.append(expl_dataset)
        return explanation_datasets

    for split in ["train", "valid", "test"]:
        expl_dataset = ExplanationDataset(
            root=root, name=cls, dataset=dataset, split=split, hops=hops
        )
        explanation_datasets.append(expl_dataset)
    return explanation_datasets


def build_dynamic_explainer_dataset(cfg, dataset, hops):
    cls = cfg.dataset["class"]
    root = cfg.dataset.root

    explanation_datasets = []

    for split in ["train", "valid", "test"]:
        expl_dataset = DynamicExplanationDataset(
            root=root, name=cls, dataset=dataset, split=split, hops=hops
        )
        data = expl_dataset[0]
        explanation_datasets.append(expl_dataset)
    return explanation_datasets


def build_explanation_dataloader(
    explanation_dataset, batch_size, num_workers, rank, world_size
):
    pw = num_workers > 0

    loaders = []
    if len(explanation_dataset) == 3:
        train_dataset, valid_dataset, test_dataset = explanation_dataset
        # partition the train, valid, test according to the world size
        index = torch.arange(len(train_dataset))
        index = torch.tensor_split(index, world_size)[rank]
        train_loader = DataLoader(
            train_dataset[index],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=pw,
        )
        loaders.append(train_loader)

        index = torch.arange(len(valid_dataset))
        index = torch.tensor_split(index, world_size)[rank]
        valid_loader = DataLoader(
            valid_dataset[index],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=pw,
        )
        loaders.append(valid_loader)

        index = torch.arange(len(test_dataset))
        index = torch.tensor_split(index, world_size)[rank]
        test_loader = DataLoader(
            test_dataset[index],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=pw,
        )
        loaders.append(test_loader)

    else:
        test_dataset = explanation_dataset[0]
        index = torch.arange(len(test_dataset))
        index = torch.tensor_split(index, world_size)[rank]
        test_loader = DataLoader(
            test_dataset[index],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=pw,
        )
        loaders.append(test_loader)

    return loaders


def create_data_structure(batched_data):
    """
    creates node_id, eval_edge_index, eval_edge_label, eval_triples for the batched data.
    """
    device = batched_data.x.device
    # construct the filter for padded triples (batch_size, padded_num_triples)
    max_subgraph_num_nodes = max(batched_data.num_nodes_subgraph)
    node_id = (
        torch.arange(max_subgraph_num_nodes)
        .expand(batched_data.num_graphs, max_subgraph_num_nodes)
        .to(device)
    )
    node_filter = node_id < batched_data.num_nodes_subgraph.unsqueeze(1)
    batched_data.node_id = node_id[node_filter]

    # construct the evaluation edges
    eval_edge_index = (
        torch.arange(batched_data.num_nodes)
        .expand((2, batched_data.num_nodes))
        .to(device)
    )
    modes = torch.repeat_interleave(batched_data.mode, batched_data.num_nodes_subgraph)
    center_nodes = torch.repeat_interleave(
        batched_data.center_node_index, batched_data.num_nodes_subgraph
    )
    # get the complement of the modes, 1 -> 0, and 0 -> 1. mode: 1 is tail batch and 0 is head batch
    modes_complement = 1 - modes  # the complement will be the index for the center node
    eval_edge_index[modes_complement, torch.arange(batched_data.num_nodes)] = (
        center_nodes
    )
    batched_data.eval_edge_index = eval_edge_index
    eval_edge_label = torch.repeat_interleave(
        batched_data.eval_rel, batched_data.num_nodes_subgraph
    ).to(device)
    batched_data.eval_edge_label = eval_edge_label

    # We have to pad the triples s.t. it becomes (batch_size, padded_num_triples, 3)
    eval_triples = torch.zeros(
        (batched_data.num_graphs, max_subgraph_num_nodes, 3), dtype=torch.int64
    ).to(device)
    eval_triples[batched_data.batch, batched_data.node_id] = torch.cat(
        [batched_data.eval_edge_index, batched_data.eval_edge_label.unsqueeze(0)]
    ).t()

    return batched_data, eval_triples


def full_graph_prediction(full_data, batch, mode, model):
    """
    Performs full graph prediction, get top k tails
    & the node and relation embeddings from the NBFNet
    Args:
        full_data: the full data
        batch: the true triples
        mode: the mode (True for tail and False for head)
        model: the explainer model (which holds the NBFNet)
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


def create_s_node_embeds(data, node_embeds):
    batch_size = node_embeds.size(0)
    # get the relevant node embeds
    s_node_embeds = torch.zeros(
        (batch_size, data.max_num_nodes, node_embeds.size(-1)),
        device=node_embeds.device,
        dtype=node_embeds.dtype,
    )
    indices = torch.arange(data.max_num_nodes, device=node_embeds.device).repeat(
        batch_size, 1
    )
    s_filter = indices < data.subgraph_num_nodes.unsqueeze(1)
    s_node_embeds[s_filter] = node_embeds[data.node_batch, data.node_id]
    return s_node_embeds


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
        node_embeds = create_s_node_embeds(data, node_embeds)
    else:
        data = data_util.prepare_full_data(train_data, len(batch), central_nodes)
        data = data_util.create_batched_data(data)  # create the batched data

    # do masking
    data = model.pred_model.masking(data, batch)
    pred, size_loss, mask_ent_loss = parallel_model(data, batch, node_embeds, epoch)
    # compute the loss only with the valid tails
    if not model.counter_factual:  # factual explanation
        label = torch.ones_like(pred)
    else:
        label = torch.zeros_like(pred)  # counter factual
    loss = F.binary_cross_entropy_with_logits(pred, label, reduction="mean")
    return loss, size_loss, mask_ent_loss


def relpgexplainer_train(
    cfg, train_data, batch, mode, model, parallel_model, epoch, is_synthetic=False
):
    """
    Training RelPGExplainer for Factual Explanation. RelPGExplainer will be optimized based on the true label.
    Args:
        cfg: the configuration
        train_data: the full training data
        batch: the true triples
        mode: the mode of the evaluation (True for tail, False for head)
        model: explainer model
        parallel_model: explainer model, could be DDP
        epoch: the epoch
        is_synthetic: whether the dataset is synthetic. Relevant for negative sampling mechanism.
    Returns:
        loss, size_loss, mask_ent_loss, rw_loss
    """
    # get the topk predicted tails and the node and rel embeddings
    topk_tails, node_embeds, R_embeds = full_graph_prediction(
        train_data, batch, mode, model
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
        node_embeds = create_s_node_embeds(data, node_embeds)
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
    preds, size_loss, mask_ent_loss, rw_loss = parallel_model(
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

    return factual_loss, counter_factual_loss, size_loss, mask_ent_loss, rw_loss


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
            neg_weight[~is_neg] = float(
                "-inf"
            )  # pos and invalid tails will get -inf to not affect the softmax
            neg_weight = F.softmax(neg_weight, dim=-1)
            neg_weight[is_pos] = 1  # pos will get 1
    else:
        num_negative = is_neg.sum(dim=-1)
        neg_weight = torch.ones_like(pred)
        neg_weight = neg_weight / num_negative.unsqueeze(1)
        neg_weight[is_pos] = 1

    neg_weight[~valid_tails] = (
        0  # the invalid tails will get a score of 0 to nullify the loss contribution
    )
    loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
    loss = loss.mean()
    return loss


def combine_stats(rank, world_size, device, stats, save_explanation, explanations):
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

    if save_explanation:
        explanations = torch.cat(explanations)
        all_explanations = torch.zeros(
            (all_size.sum(), explanations.size(1)),
            dtype=explanations.dtype,
            device="cpu",
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
):
    scores = {}
    if rank == 0:
        eval_type_info = ""
        additional_info = ""
        if eval_type is not None:
            eval_type_info += f"_{eval_type}"
            additional_info += f"_{eval_type}"
        if eval_mask_type is not None:
            additional_info += f"_{eval_mask_type}_{ratio}"
        stats = {}
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
        if working_dir is not None:
            data = {
                "Ranking": all_stats["ranking"].tolist(),
                "Heads": all_stats["heads"].tolist(),
                "Tails": all_stats["tails"].tolist(),
                "Rel": all_stats["rels"].tolist(),
                "Mode": all_stats["modes"].tolist(),
                "Num_Edges": all_stats["num_edges"].tolist(),
                "Num_Nodes": all_stats["num_nodes"].tolist(),
                "Inclusion": all_stats["inclusion"].tolist(),
            }
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
