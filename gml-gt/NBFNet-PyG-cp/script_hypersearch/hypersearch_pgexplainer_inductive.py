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
    cfg.explainer.size_reg = random.randint(1, 5)
    cfg.explainer.ent_reg = random.randint(1, 5)
    cfg.explainer.sample_bias = random.choice([0.1, 0.3, 0.5])
    cfg.explainer.temp_start = random.randint(1, 5)
    cfg.explainer.temp_end = random.randint(1, cfg.explainer.temp_start) # should be lower than temp start
    cfg.explainer.num_mlp_layer = random.randint(2, 4)
    cfg.optimizer.lr = random.choice([1e-3, 1e-4, 1e-5, 5e-3, 5e-4, 5e-5])
    cfg.explainer.ego_network = random.choice([True, False])
    cfg.wandb.name = cfg.wandb.name+f"_{args.hyper_run_id}"
    torch.save(cfg, f'hyperparam_cfg_pgex_inductive.pt')

    