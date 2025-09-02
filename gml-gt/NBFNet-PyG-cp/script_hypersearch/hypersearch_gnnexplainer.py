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
    cfg.explainer.size_reg = random.randint(1, 4)
    cfg.explainer.ent_reg = random.randint(1, 5)
    cfg.explainer.topk_tails = random.randint(1, 3)
    cfg.explainer.ego_network = random.choice([True, False])
    cfg.train.num_epoch = random.choice([10, 20, 30, 40, 50])
    cfg.optimizer.lr = random.choice([1e-2, 5e-3, 1e-3, 5e-4])
    cfg.wandb.name = cfg.wandb.name+f"_{args.hyper_run_id}"
    torch.save(cfg, f'hyperparam_cfg_gnnexplainer.pt')

    