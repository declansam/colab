import os
import sys
import random
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import util

separator = ">" * 30
line = "-" * 30

if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    # cfg.explainer.topk_tails = random.randint(1, 3)
    cfg.explainer.prune_max_degree = random.choice([30, 50, 100])
    cfg.explainer.k_core = random.choice([2, 3, 4])
    cfg.explainer.num_paths = random.choice([2, 4, 6])
    cfg.explainer.alpha = random.choice([0.5, 1.0, 1.5, 2.0])
    cfg.explainer.beta = random.choice([0.5, 1.0, 1.5, 2.0])
    cfg.train.num_epoch = random.choice([2, 4, 6])
    cfg.optimizer.lr = random.choice([1e-2, 5e-3, 1e-3, 5e-4])
    cfg.wandb.name = cfg.wandb.name+f"_{args.hyper_run_id}"
    torch.save(cfg, f'hyperparam_cfg_pagelink.pt')

    