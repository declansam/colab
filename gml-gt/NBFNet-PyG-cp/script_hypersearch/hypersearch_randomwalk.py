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
    cfg.explainer.topk_tails = random.randint(1, 3)
    cfg.explainer.ego_network = random.choice([True, False])
    # RW params
    cfg.explainer.adj_aggr = random.choice(["max", "mean", "sum"])
    cfg.explainer.teleport_prob = random.choice([0.001, 0.01, 0.05, 0.1, 0.15, 0.2])
    cfg.explainer.use_teleport_adj = random.choice([True, False])
    cfg.wandb.name = cfg.wandb.name + f"_{args.hyper_run_id}"
    torch.save(cfg, f"hyperparam_cfg_randomwalk.pt")
