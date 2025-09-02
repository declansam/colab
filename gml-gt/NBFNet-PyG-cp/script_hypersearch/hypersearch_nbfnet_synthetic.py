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
    cfg.model.input_dim = random.choice([16, 24, 32, 48])
    # hops = random.choice
    cfg.model.hidden_dims = [cfg.model.input_dim, cfg.model.input_dim]
    cfg.optimizer.lr = random.choice([1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5])

    cfg.wandb.name = cfg.wandb.name+f"_{args.hyper_run_id}"
    torch.save(cfg, f'hyperparam_cfg_rgcn.pt')

    