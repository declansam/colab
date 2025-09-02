import os
import sys
import pprint
from itertools import chain
import torch
import atexit
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
from torch import optim
import torch.multiprocessing as mp
from tqdm import tqdm
import math

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import tasks, util
from explainers import explainer_util

separator = ">" * 30
line = "-" * 30

        
def train_explainer(cfg, explainer, logger, train_loader, valid_loader, test_loader, device):
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
    
    util.synchronize()

    cls = cfg.optimizer.pop("class")
    # optimizer will only change the parameter of explainer, not NBFNet.
    optimizer = getattr(optim, cls)(filter(lambda p: p.requires_grad, explainer.parameters()), **cfg.optimizer)

    if world_size > 1:
        parallel_explainer = DistributedDataParallel(explainer, device_ids=[device])
    else:
        parallel_explainer = explainer

    if hasattr(cfg.train, "step"):
        step = cfg.train.step
    else:
        step = math.ceil(cfg.train.num_epoch / 10)

    best_result = float("-inf")
    best_epoch = -1

    for epoch in range(0, cfg.train.num_epoch):
        parallel_explainer.train()

        if explainer.return_detailed_loss:
            bce_losses = []
            size_losses = []
            mask_ent_losses = []
            losses = []
        else:
            losses = []

        if rank == 0:
            pbar = tqdm(total=len(train_loader), desc='Processing Batch')

        for iter, batched_data in enumerate(train_loader):
            batched_data.to(device)

            batched_data, eval_triples = explainer_util.create_data_structure(batched_data)

            loss = parallel_explainer(batched_data, eval_triples, epoch)

            if explainer.return_detailed_loss:
                bce_loss, size_loss, mask_ent_loss = loss
                bce_losses.append(bce_loss.item())
                size_losses.append(size_loss.item())
                mask_ent_losses.append(mask_ent_loss.item())
                loss = bce_loss + size_loss + mask_ent_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # if rank == 0:
            #     logger.warning(separator)
            #     logger.warning("binary cross entropy: %g" % loss)

            losses.append(loss.item())

            if rank == 0:
                pbar.update(1)
        
        if rank == 0:
            pbar.close()
            avg_loss = sum(losses) / len(losses)
            logger.warning(separator)
            logger.warning("Epoch %d end" % epoch)
            logger.warning(line)
            logger.warning("average loss: %g" % avg_loss)

            if cfg.wandb.use:
                if epoch%step == 0:
                    commit = False
                else:
                    commit = True
                stats = {'train/loss': avg_loss}
                if explainer.return_detailed_loss:
                    stats.update({'train/bce_loss': sum(bce_losses)/len(bce_losses),
                                  'train/size_loss': sum(size_losses)/len(size_losses),
                                  'train/mask_ent_loss': sum(mask_ent_losses)/len(mask_ent_losses)
                                  })
                run.log(stats, step=epoch, commit=commit)

        if epoch%step == 0:
            if rank == 0:
                logger.warning("Save checkpoint to explainer_epoch_%d.pth" % epoch)
                state = {
                    "explainer": explainer.state_dict(),
                    "optimizer": optimizer.state_dict()
                }
                torch.save(state, "explainer_epoch_%d.pth" % epoch)

                logger.warning(separator)
                logger.warning("Evaluate on valid")

            util.synchronize()

            result = test_explainer(cfg, explainer, logger, valid_loader, device, 'valid', run, epoch)
            if rank == 0:
                logger.warning(separator)
                logger.warning("Evaluate on test")
            test_explainer(cfg, explainer, logger, test_loader, device, 'test', run, epoch)

        if result > best_result:
            best_result = result
            best_epoch = epoch

    if rank == 0:
        logger.warning("Load checkpoint from explainer_epoch_%d.pth" % best_epoch)
    state = torch.load("explainer_epoch_%d.pth" % best_epoch, map_location=device)
    explainer.load_state_dict(state["explainer"])
    util.synchronize()
    if cfg.wandb.use and rank==0:
        run.finish()

@torch.no_grad()
def test_explainer(cfg, explainer, logger, loader, device, split_name, run=None, epoch=None):
    world_size = util.get_world_size()
    rank = util.get_rank()

    explainer.eval()
    rankings = []
    modes = [] # logging the mode (1 for tail, 0 for head)
    rels = [] # logging the rel types.
    inclusions = []

    if rank == 0:
        pbar = tqdm(total=len(loader), desc='Processing Batch')

    for iter, batched_data in enumerate(loader):
        batched_data.to(device)
        
        batched_data, eval_triples = explainer_util.create_data_structure(batched_data)
        
        ranking, inclusion = explainer(batched_data, eval_triples)
        rankings.append(ranking)
        inclusions.append(inclusion)
        rels.append(batched_data.eval_rel)
        modes.append(batched_data.mode)
        if rank == 0:
            pbar.update(1)

    if rank == 0:
        pbar.close()

    ranking = torch.cat(rankings)
    inclusion = torch.cat(inclusions)
    rels = torch.cat(rels)
    modes = torch.cat(modes)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
    cum_size = all_size.cumsum(0)

    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_inclusion = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_inclusion[cum_size[rank] - all_size[rank]: cum_size[rank]] = inclusion
    all_rels = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_rels[cum_size[rank] - all_size[rank]: cum_size[rank]] = rels
    all_modes = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_modes[cum_size[rank] - all_size[rank]: cum_size[rank]] = modes

    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_inclusion, op=dist.ReduceOp.SUM)
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
                score = (all_ranking <= threshold).float().mean()
            elif metric == "inclusion":
                score = all_inclusion.float().mean()
            stats[f'{split_name}/{metric}'] = score.item()
            logger.warning("%s: %g" % (metric, score))
        
        if cfg.wandb.use:
            if split_name == 'test':
                commit = True
            else:
                commit = False
            run.log(stats, step=epoch, commit=commit)

    mrr = (1 / all_ranking.float()).mean()

    return mrr

def run(rank, world_size, cfg, args, explainer_dataset):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    # load the model from checkpoint
    cfg.model.num_relation = explainer_dataset[0].num_relations
    model = util.build_model(cfg)
    device = util.get_device(cfg)
    model = model.to(device)

    loaders = explainer_util.build_explanation_dataloader(explainer_dataset, cfg.train.batch_size, cfg.train.num_workers, rank, world_size)
    # initialize the explainer
    explainer = explainer_util.build_explainer(cfg, model)
    explainer = explainer.to(device)

    if len(loaders) == 3:
        train_explainer(cfg, explainer, logger, loaders[0], loaders[1], loaders[2], device)
        test_loader = loaders[2]
    else:
        test_loader = loaders[0]
    # disable wandb
    cfg.wandb.use = False
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")
    
    test_explainer(cfg, explainer, logger, test_loader, device, 'test')

    dist.destroy_process_group()

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

atexit.register(cleanup)

if __name__ == "__main__":
    # load the config, and set up the experiment
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)

    # prepare the dataset
    dataset = util.build_dataset(cfg)
    dataset.id2entity, dataset.id2relation = util.load_id2name(dataset, cfg.dataset["class"])

    # load the model from checkpoint, we need to know the number of hops
    cfg.model.num_relation = dataset.num_relations
    model = util.build_model(cfg)

    if hasattr(cfg, 'evaluation'):
        evaluation = cfg.evaluation
    else:
        evaluation = None

    explainer_dataset = explainer_util.build_explainer_dataset(cfg, dataset, evaluation, model.hops)

    world_size = util.get_world_size(len(cfg.train.gpus))

    mp.spawn(run, args=(world_size, cfg, args, explainer_dataset), nprocs=world_size, join=True)