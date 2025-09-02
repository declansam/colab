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
    # Explainer Params
    cfg.explainer.size_reg = random.randint(3, 7)
    cfg.explainer.ent_reg = random.randint(2, 5)
    cfg.explainer.sample_bias = random.choice([0.3, 0.4, 0.5])
    cfg.explainer.temp_start = random.randint(1, 5)
    # should be lower than temp start
    cfg.explainer.temp_end = random.randint(1, cfg.explainer.temp_start)
    cfg.explainer.num_mlp_layer = random.randint(2, 4)
    cfg.explainer.use_default_aggr = random.choice([False])
    cfg.explainer.topk_tails = random.randint(1, 3)
    cfg.explainer.ego_network = random.choice([True, False])
    cfg.optimizer.lr = random.choice([1e-4, 5e-4, 1e-5, 5e-5])
    # RW params
    # cfg.explainer.random_walk_loss = random.choice([True, False])
    cfg.explainer.random_walk_loss = True
    cfg.explainer.adj_aggr = random.choice(["max", "mean", "sum"])
    cfg.explainer.teleport_prob = random.choice([0.05, 0.1, 0.15, 0.2])
    cfg.explainer.rw_topk_node = random.choice([100, 200, 300, 500, 1000])
    cfg.explainer.reg_loss_inside = random.choice([1, 1.5, 2, 2.5, 3])
    cfg.explainer.reg_loss_outside = random.choice([0, 0.5, 1, 1.5, 2])
    cfg.explainer.use_teleport_adj = random.choice([True])
    # Path Params
    cfg.explainer.with_path_loss = random.choice([True, False])
    cfg.explainer.reg_path_loss = random.randint(1, 5)
    cfg.explainer.max_path_length = 3
    # Optim Params
    cfg.explainer.factual = random.choice([True, False])
    # cfg.explainer.factual = True
    if not cfg.explainer.factual:  # must be either factual or counter factual
        cfg.explainer.counter_factual = True
    else:
        cfg.explainer.counter_factual = random.choice([True, False])
        # cfg.explainer.counter_factual = True
    cfg.explainer.expl_gnn_model = random.choice([True, False])
    # cfg.explainer.expl_gnn_model = True

    cfg.wandb.name = cfg.wandb.name + f"_{args.hyper_run_id}"
    torch.save(cfg, f"hyperparam_cfg_rawexplainer.pt")
