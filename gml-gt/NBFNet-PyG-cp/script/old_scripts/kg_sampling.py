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

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import tasks, util
from explainers import explainer_util
from explainers import data_util

separator = ">" * 30
line = "-" * 30


@torch.no_grad()
def full_graph_prediction(full_data, batch, mode, model):
    """
    Full Graph Prediction, get top k tails & the node and relation embeddings from the NBFNet
    """
    batch_size = len(batch)
    t_batch, h_batch = tasks.all_negative(full_data, batch)
    batch = torch.where(mode.view(batch_size, 1, 1), t_batch, h_batch)
    # prepare the ego data based on the heads
    data = data_util.prepare_full_data(full_data, batch_size)
    # convert the batch to tail prediction
    batch = tasks.conversion_to_tail_prediction(
        batch, model.model_to_explain.num_relation, mode
    )
    # do masking
    data = model.model_to_explain.masking(data, batch)
    # create batched_edge_index and edge_filter based on the masked data
    data = data_util.create_batched_data(data)

    original_pred, embeds, R = model.full_prediction(data, batch)
    tails = torch.argsort(original_pred, dim=1, descending=True)
    # option to filter out the other true tails, during training, you can also put the true tail
    return tails[:, : model.topk_tails], embeds, R


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


def train_and_validate(
    cfg, model, train_data, valid_data, logger, device, filtered_data=None
):
    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()

    run = util.wandb_setup(cfg, rank)

    train_triplets = torch.cat(
        [train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0)]
    ).t()
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
    train_loader = torch_data.DataLoader(
        train_triplets, cfg.train.batch_size, sampler=sampler
    )

    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)
    # optimizer = getattr(optim, cls)(filter(lambda p: p.requires_grad, model.parameters()), **cfg.optimizer)

    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

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

        bce_losses = []
        size_losses = []
        mask_ent_losses = []
        losses = []
        sampler.set_epoch(epoch)

        if rank == 0:
            pbar = tqdm(total=len(train_loader), desc="Processing Batch")

        for batch in train_loader:  # batch: (batch_size, 3)
            batch_size = len(batch)
            # create the mode and the heads respectivelly
            mode = torch.zeros((batch_size,), dtype=torch.bool, device=batch.device)
            mode[: batch_size // 2] = (
                1  # first half will be tail batch, and the second half will be head batch
            )

            # get the topk predicted tails and the node and rel embeddings
            topk_tails, node_embeds, R_embeds = full_graph_prediction(
                train_data, batch, mode, model
            )
            # create the central nodes for making the ego network
            heads = torch.where(mode, batch[:, 0], batch[:, 1])
            central_nodes = torch.cat((heads.unsqueeze(1), topk_tails), dim=-1)
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
            )
            # convert the batch to tail prediction
            batch = tasks.conversion_to_tail_prediction(
                batch, model.model_to_explain.num_relation, mode
            )
            # relabel the batch and also get the mask indicating the tails that are inside the ego network
            batch, valid_tails = data_util.relabel_batch(data, batch)
            # do masking
            data = model.model_to_explain.masking(data, batch)
            # create batched_edge_index and edge_filter based on the masked data
            data = data_util.create_batched_data(data)
            s_node_embeds = create_s_node_embeds(data, node_embeds)

            pred, size_loss, mask_ent_loss = parallel_model(
                data, batch, s_node_embeds, R_embeds, epoch
            )

            is_pos = torch.zeros_like(pred, dtype=torch.bool)
            is_pos[:, 0] = 1
            # sometimes the pos tail is not included in the ego network
            is_pos[~pos_included] = 0
            is_neg = ~is_pos
            # sometimes there is no neg tail to sample from the ego_network (singleton)
            is_neg[~valid_tails] = 0
            # check there are no mismatchs
            assert torch.all(
                is_pos.to(torch.long)
                + is_neg.to(torch.long)
                + (~valid_tails).to(torch.long)
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

            bce_losses.append(loss.item())
            size_losses.append(size_loss.item())
            mask_ent_losses.append(mask_ent_loss.item())

            loss = loss + size_loss + mask_ent_loss

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
            # if batch_id == 5:
            #     break

        if util.get_rank() == 0:
            avg_loss = sum(losses) / len(losses)
            logger.warning(separator)
            logger.warning("Epoch %d end" % epoch)
            logger.warning(line)
            logger.warning("average binary cross entropy: %g" % avg_loss)
            pbar.close()
            if cfg.wandb.use:
                if epoch % step == 0 or epoch == cfg.train.num_epoch - 1:
                    commit = False
                else:
                    commit = True
                stats = {
                    "train/bce_loss": sum(bce_losses) / len(bce_losses),
                    "train/size_loss": sum(size_losses) / len(size_losses),
                    "train/mask_ent_loss": sum(mask_ent_losses) / len(mask_ent_losses),
                    "train/loss": avg_loss,
                }
                run.log(stats, step=epoch, commit=commit)

        if rank == 0:
            logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(state, "model_epoch_%d.pth" % epoch)
        util.synchronize()

        if epoch % step == 0 or epoch == cfg.train.num_epoch - 1:
            if "hard" in cfg.explainer_eval.eval_mask_type:
                ratios = getattr(cfg.explainer_eval, cfg.explainer_eval.eval_mask_type)
                results = []
                for i, r in enumerate(ratios):
                    if rank == 0:
                        logger.warning(separator)
                        logger.warning(
                            f"Evaluate on valid with {cfg.explainer_eval.eval_mask_type}: {r}"
                        )
                    result = test(
                        cfg,
                        model,
                        valid_data,
                        logger,
                        device,
                        filtered_data=filtered_data,
                        split="valid",
                        run=run,
                        epoch=epoch,
                        eval_mask_type=cfg.explainer_eval.eval_mask_type,
                        ratio=r,
                        commit=False,
                    )
                    results.append(result)
                result = sum(results) / len(results)
                if rank == 0:
                    stats = {"valid/avg_mrr": result}
                    run.log(
                        stats, step=epoch, commit=True
                    )  # log the average mrr score across validation sets

            else:
                result = test(
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

            if result > best_result:
                best_result = result
                best_epoch = epoch

    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device)
    model.load_state_dict(state["model"])
    util.synchronize()


@torch.no_grad()
def test_prediction(test_data, triples, mode, model, batch):
    # get the topk predicted tails and the node and rel embeddings
    topk_tails, node_embeds, R_embeds = full_graph_prediction(
        test_data, triples, mode, model
    )
    heads = torch.where(mode, triples[:, 0], triples[:, 1])
    central_nodes = torch.cat((heads.unsqueeze(1), topk_tails), dim=-1)
    # prepare the ego data based on the central nodes
    data = data_util.prepare_ego_data(test_data, central_nodes, model.hops)
    batch = tasks.conversion_to_tail_prediction(
        batch, model.model_to_explain.num_relation, mode
    )
    batch = data_util.check_if_tail_in_network(data, batch)
    batch, valid_tails = data_util.relabel_batch(data, batch)
    # do masking
    data = model.model_to_explain.masking(data, batch)
    # create batched_edge_index and edge_filter based on the masked data
    data = data_util.create_batched_data(data)
    s_node_embeds = create_s_node_embeds(data, node_embeds)

    pred, node_mask, num_edges, num_nodes = model(data, batch, s_node_embeds, R_embeds)
    pred[~valid_tails] = float(
        "-inf"
    )  # for the predictions outside the ego network, it will get a score of -inf
    pred[~node_mask] = float(
        "-inf"
    )  # for the predictions outside the explanatory subgraph, it will get a score of -inf
    return pred, node_mask, num_edges, num_nodes


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
    eval_mask_type=None,
    ratio=None,
    commit=True,
):
    world_size = util.get_world_size()
    rank = util.get_rank()

    test_triplets = torch.cat(
        [test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]
    ).t()
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(
        test_triplets, cfg.train.batch_size, sampler=sampler
    )

    if rank == 0:
        pbar = tqdm(total=len(test_loader))

    model.eval()

    if eval_mask_type is not None:
        setattr(model, eval_mask_type, ratio)

    rankings = []
    inclusions = []
    num_negatives = []
    num_edges = []
    num_nodes = []
    modes = []  # logging the mode (0 for tail, 1 for head)
    rels = []  # logging the rel types.

    # count = 0
    for batch in test_loader:
        batch_size = len(batch)
        t_batch, h_batch = tasks.all_negative(test_data, batch)

        # TAIL PREDICTION FIRST
        mode = torch.ones((batch_size,), dtype=torch.bool, device=batch.device)
        t_pred, t_node_mask, t_num_edges, t_num_nodes = test_prediction(
            test_data, triples=batch, mode=mode, model=model, batch=t_batch
        )

        # HEAD PREDICTION NEXT
        mode = torch.zeros((batch_size,), dtype=torch.bool, device=batch.device)
        h_pred, h_node_mask, h_num_edges, h_num_nodes = test_prediction(
            test_data, triples=batch, mode=mode, model=model, batch=h_batch
        )

        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)

        pos_h_index, pos_t_index, pos_r_index = batch.t()
        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
        t_inclusion = t_node_mask[torch.arange(batch_size), pos_t_index]
        h_inclusion = h_node_mask[torch.arange(batch_size), pos_h_index]
        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)

        rankings += [t_ranking, h_ranking]
        inclusions += [t_inclusion, h_inclusion]
        num_negatives += [num_t_negative, num_h_negative]
        num_edges += [t_num_edges, h_num_edges]
        num_nodes += [t_num_nodes, h_num_nodes]
        rels += [
            batch[:, -1],
            batch[:, -1],
        ]  # get the rels for both tail and head predictions
        modes += [
            torch.zeros(t_ranking.shape),
            torch.ones(h_ranking.shape),
        ]  # log whether it was for tail prediction or head prediction

        if rank == 0:
            pbar.update(1)
        # count+=1
        # if count == 10:
        #     break

    if rank == 0:
        pbar.close()

    ranking = torch.cat(rankings)
    inclusion = torch.cat(inclusions)
    num_negative = torch.cat(num_negatives)
    num_edges = torch.cat(num_edges)
    num_nodes = torch.cat(num_nodes)
    rels = torch.cat(rels)
    modes = torch.cat(modes)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank] : cum_size[rank]] = ranking
    all_inclusion = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_inclusion[cum_size[rank] - all_size[rank] : cum_size[rank]] = inclusion
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank] : cum_size[rank]] = num_negative
    all_num_edges = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_edges[cum_size[rank] - all_size[rank] : cum_size[rank]] = num_edges
    all_num_nodes = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_nodes[cum_size[rank] - all_size[rank] : cum_size[rank]] = num_nodes
    all_rels = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_rels[cum_size[rank] - all_size[rank] : cum_size[rank]] = rels
    all_modes = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_modes[cum_size[rank] - all_size[rank] : cum_size[rank]] = modes

    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_inclusion, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_edges, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_nodes, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_rels, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_modes, op=dist.ReduceOp.SUM)

    if rank == 0:
        stats = {}
        for metric in cfg.task.metric:
            if metric == "mr":
                score = all_ranking.float().mean()
            elif metric == "mrr":
                score = (1 / all_ranking.float()).mean()
            elif metric.startswith("hits@"):
                values = metric[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    # unbiased estimation
                    fp_rate = (all_ranking - 1).float() / all_num_negative
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
                    score = (all_ranking <= threshold).float().mean()
            elif metric == "inclusion":
                score = all_inclusion.float().mean()
            elif metric == "num_edges":
                score = all_num_edges.float().mean()
            elif metric == "num_nodes":
                score = all_num_nodes.float().mean()
            additional_info = ""
            if eval_mask_type is not None:
                additional_info = f"_{eval_mask_type}_{ratio}"
            stats[f"{split}/{metric}{additional_info}"] = score.item()
            logger.warning("%s: %g" % (metric + additional_info, score))

        if cfg.wandb.use:
            run.log(stats, step=epoch, commit=commit)

        if working_dir is not None:
            data = {
                "Ranking": all_ranking.tolist(),
                "Rel": all_rels.tolist(),
                "Mode": all_modes.tolist(),
            }
            torch.save(data, os.path.join(working_dir, f"{split}_output.pt"))

    mrr = (1 / all_ranking.float()).mean()

    return mrr


def run(cfg, args):
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
    is_inductive = cfg.dataset["class"].startswith("Ind")
    dataset = util.build_dataset(cfg)

    cfg.model.num_relation = dataset.num_relations
    model = util.build_model_expl(cfg)

    explainer = explainer_util.build_ego_explainer(cfg, model)

    device = util.get_device(cfg)
    model = model.to(device)
    explainer = explainer.to(device)
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)
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

    if "evaluate_on_train" in cfg and cfg.evaluate_on_train:
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning("Evaluate on train")
        test(
            cfg,
            explainer,
            train_data,
            logger,
            device,
            filtered_data=filtered_data,
            working_dir=working_dir,
            split="train",
        )

    if "hard" in cfg.explainer_eval.eval_mask_type:
        ratios = getattr(cfg.explainer_eval, cfg.explainer_eval.eval_mask_type)
        results = []
        for r in ratios:
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning(
                    f"Evaluate on valid with {cfg.explainer_eval.eval_mask_type}: {r}"
                )
            test(
                cfg,
                explainer,
                valid_data,
                logger,
                device,
                filtered_data=filtered_data,
                working_dir=working_dir,
                split="valid",
                eval_mask_type=cfg.explainer_eval.eval_mask_type,
                ratio=r,
            )
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning(
                    f"Evaluate on test with {cfg.explainer_eval.eval_mask_type}: {r}"
                )
            test(
                cfg,
                explainer,
                test_data,
                logger,
                device,
                filtered_data=filtered_data,
                working_dir=working_dir,
                split="test",
                eval_mask_type=cfg.explainer_eval.eval_mask_type,
                ratio=r,
            )
    else:
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        test(
            cfg,
            explainer,
            valid_data,
            logger,
            device,
            filtered_data=filtered_data,
            working_dir=working_dir,
            split="valid",
        )
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning("Evaluate on test")
        test(
            cfg,
            explainer,
            test_data,
            logger,
            device,
            filtered_data=filtered_data,
            working_dir=working_dir,
            split="test",
        )


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    if args.hyper_search:
        cfg = torch.load("hyperparam_cfg.pt")
        run(cfg, args)
    else:
        run(cfg, args)
