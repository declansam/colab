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


separator = ">" * 30
line = "-" * 30


def train_and_validate(cfg, model, train_data, valid_data, filtered_data=None):
    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()

    if cfg.wandb.use and rank==0:
        try:
            import wandb
        except:
            raise ImportError('WandB is not installed.')
        wandb_name = cfg.wandb.name
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                         name=wandb_name)
        run.config.update(cfg)
    else:
        run = None

    train_triplets = torch.cat([train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
    train_loader = torch_data.DataLoader(train_triplets, cfg.train.batch_size, sampler=sampler)

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
            pbar = tqdm(total=len(train_loader), desc='Processing Batch')

        for batch in train_loader: # batch: (batch_size, 3)
            batch = tasks.negative_sampling(train_data, batch, cfg.task.num_negative,
                                            strict=cfg.task.strict_negative)
            pred, size_loss, mask_ent_loss = parallel_model(train_data, batch, epoch)
            target = torch.zeros_like(pred)
            target[:, 0] = 1
            loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            neg_weight = torch.ones_like(pred)
            if cfg.task.adversarial_temperature > 0:
                with torch.no_grad(): # the higher the score is for a negative pred, the more it will contribute
                    neg_weight[:, 1:] = F.softmax(pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1)
            else:
                neg_weight[:, 1:] = 1 / cfg.task.num_negative
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
            # if batch_id == 10:
            #     break

        if util.get_rank() == 0:
            avg_loss = sum(losses) / len(losses)
            logger.warning(separator)
            logger.warning("Epoch %d end" % epoch)
            logger.warning(line)
            logger.warning("average binary cross entropy: %g" % avg_loss)
            pbar.close()
            if cfg.wandb.use:
                if epoch%step == 0 or epoch == cfg.train.num_epoch-1:
                    commit = False
                else:
                    commit = True
                stats = {'train/bce_loss': sum(bce_losses)/len(bce_losses),
                        'train/size_loss': sum(size_losses)/len(size_losses),
                        'train/mask_ent_loss': sum(mask_ent_losses)/len(mask_ent_losses),
                        'train/loss': avg_loss
                        }
                run.log(stats, step=epoch, commit=commit)

        if rank == 0:
            logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, "model_epoch_%d.pth" % epoch)
        util.synchronize()

        if epoch%step == 0 or epoch == cfg.train.num_epoch-1:
            if "hard" in cfg.explainer_eval.eval_mask_type:
                ratios = getattr(cfg.explainer_eval, cfg.explainer_eval.eval_mask_type)
                results = []
                for i, r in enumerate(ratios):
                    if i == len(ratios)-1:
                        commit = True
                    else:
                        commit = False
                    if rank == 0:
                        logger.warning(separator)
                        logger.warning(f"Evaluate on valid with {cfg.explainer_eval.eval_mask_type}: {r}")
                    result = test(cfg, model, valid_data, filtered_data=filtered_data, split='valid', run=run, epoch=epoch, eval_mask_type=cfg.explainer_eval.eval_mask_type, ratio=r, commit=commit)
                    results.append(result)
                result = sum(results)/len(results)

            else:
                result = test(cfg, model, valid_data, filtered_data=filtered_data, split='valid', run=run, epoch=epoch)
                
            if result > best_result:
                best_result = result
                best_epoch = epoch

    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device)
    model.load_state_dict(state["model"])
    util.synchronize()


@torch.no_grad()
def test(cfg, model, test_data, filtered_data=None, working_dir=None, split='test', run=None, epoch=None, eval_mask_type=None, ratio=None, commit=True, save_explanation=False):
    world_size = util.get_world_size()
    rank = util.get_rank()

    if save_explanation:
        assert world_size == 1

    test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

    if rank == 0:
        pbar = tqdm(total=len(test_loader))

    model.eval()

    if eval_mask_type is not None:
        setattr(model, eval_mask_type, ratio)

    rankings = []
    num_negatives = []
    modes = [] # logging the mode (1 for tail, 0 for head)
    heads = []
    rels = []
    tails = []
    explanations = []

    # count = 0
    for batch in test_loader:
        t_batch, h_batch = tasks.all_negative(test_data, batch)
        t_pred = model(test_data, t_batch, save_explanation=save_explanation) # t_batch: (batch_size, num_nodes, 3)
        h_pred = model(test_data, h_batch, save_explanation=save_explanation)

        if save_explanation:
            t_pred, t_explanation = t_pred
            h_pred, h_explanation = h_pred

        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)

        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

        heads += [batch[:, 0], batch[:, 1]]
        tails += [batch[:, 1], batch[:, 0]]
        rels += [batch[:, -1], batch[:, -1]] # get the rels for both tail and head predictions

        modes += [torch.ones(t_ranking.shape), torch.zeros(h_ranking.shape)] # log whether it was for tail prediction or head prediction
        if save_explanation:
            explanations += [t_explanation.to('cpu'), h_explanation.to('cpu')]

        if rank == 0:
            pbar.update(1)
        # count+=1
        # if count == 10:
        #     break
    
    if rank == 0:
        pbar.close()

    ranking = torch.cat(rankings)
    num_negative = torch.cat(num_negatives)
    heads = torch.cat(heads)
    tails = torch.cat(tails)
    rels = torch.cat(rels)
    modes = torch.cat(modes)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative
    all_rels = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_rels[cum_size[rank] - all_size[rank]: cum_size[rank]] = rels
    all_heads = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_heads[cum_size[rank] - all_size[rank]: cum_size[rank]] = heads
    all_tails = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_tails[cum_size[rank] - all_size[rank]: cum_size[rank]] = tails
    all_modes = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_modes[cum_size[rank] - all_size[rank]: cum_size[rank]] = modes

    if save_explanation:
        explanations = torch.cat(explanations)
        all_explanations = torch.zeros((all_size.sum(), explanations.size(1)), dtype=explanations.dtype, device='cpu')
        all_explanations[cum_size[rank]-all_size[rank]:cum_size[rank], :] = explanations
    

    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_heads, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_tails, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_rels, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_modes, op=dist.ReduceOp.SUM)
        if save_explanation:
            dist.all_reduce(all_explanations, op=dist.ReduceOp.SUM)

    if rank == 0:
        additional_info = ''
        if eval_mask_type is not None:
            additional_info = f'_{eval_mask_type}_{ratio}'

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
                        num_comb = math.factorial(num_sample - 1) / \
                                   math.factorial(i) / math.factorial(num_sample - i - 1)
                        score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                    score = score.mean()
                else:
                    score = (all_ranking <= threshold).float().mean()
            stats[f'{split}/{metric}{additional_info}'] = score.item()
            logger.warning("%s: %g" % (metric+additional_info, score))
        
        if cfg.wandb.use:
            run.log(stats, step=epoch, commit=commit)
        
        if working_dir is not None:
            data = {'Ranking': all_ranking.tolist(),
                    'Heads': all_heads.tolist(),
                    'Tails': all_tails.tolist(),
                    'Rel': all_rels.tolist(),
                    'Mode': all_modes.tolist()
                    }
            torch.save(data, os.path.join(working_dir, f'{split}_output{additional_info}.pt'))
            if save_explanation:
                torch.save(all_explanations, os.path.join(working_dir, f'{split}_explanations{additional_info}.pt'))

    mrr = (1 / all_ranking.float()).mean()

    return mrr


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
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
    model = util.build_model(cfg)
    explainer = explainer_util.build_explainer(cfg, model)


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
        filtered_data = Data(edge_index=dataset._data.target_edge_index, edge_type=dataset._data.target_edge_type, num_nodes=train_data.num_nodes)
        filtered_data = filtered_data.to(device)

    train_and_validate(cfg, explainer, train_data, valid_data, filtered_data=filtered_data)

    # disable wandb
    cfg.wandb.use = False

    if "evaluate_on_train" in cfg and cfg.evaluate_on_train:
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning("Evaluate on train")
        test(cfg, explainer, train_data, filtered_data=filtered_data, working_dir=working_dir, split='train')

    if "save_explanation" in cfg and cfg.save_explanation:
        save_explanation = True
    else:
        save_explanation = False

    if "hard" in cfg.explainer_eval.eval_mask_type:
        ratios = getattr(cfg.explainer_eval, cfg.explainer_eval.eval_mask_type)
        results = []
        for r in ratios:
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning(f"Evaluate on valid with {cfg.explainer_eval.eval_mask_type}: {r}")
            test(cfg, explainer, valid_data, filtered_data=filtered_data, working_dir=working_dir, split='valid', eval_mask_type=cfg.explainer_eval.eval_mask_type, ratio=r, save_explanation=save_explanation)
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning(f"Evaluate on test with {cfg.explainer_eval.eval_mask_type}: {r}")
            test(cfg, explainer, test_data, filtered_data=filtered_data, working_dir=working_dir, split='test',  eval_mask_type=cfg.explainer_eval.eval_mask_type, ratio=r, save_explanation=save_explanation)
    else:
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        test(cfg, explainer, valid_data, filtered_data=filtered_data, working_dir=working_dir, split='valid', save_explanation=save_explanation)
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning("Evaluate on test")
        test(cfg, explainer, test_data, filtered_data=filtered_data, working_dir=working_dir, split='test', save_explanation=save_explanation)