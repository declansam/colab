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
    cfg.explainer.size_reg = random.randint(3, 5)
    cfg.explainer.ent_reg = random.randint(2, 5)
    cfg.explainer.sample_bias = random.choice([0.3, 0.4, 0.5])
    cfg.explainer.temp_start = random.randint(1, 5)
    # should be lower than temp start
    cfg.explainer.temp_end = random.randint(1, cfg.explainer.temp_start)
    cfg.explainer.num_mlp_layer = random.randint(2, 3)
    cfg.explainer.topk_tails = random.randint(2, 4)
    cfg.explainer.ego_network = random.choice([True, False])
    # Optim, num_epoch, neg samples
    cfg.optimizer.lr = random.choice([1e-4, 2.5e-4, 5e-4, 1e-5, 5e-5])
    cfg.train.num_epoch = random.choice([5])
    cfg.task.num_negative = random.choice([0, 32])
    # RW params
    # cfg.explainer.random_walk_loss = random.choice([True, False])
    cfg.explainer.random_walk_loss = False
    cfg.explainer.teleport_prob = random.choice([0.15, 0.175, 0.2, 0.225, 0.25])
    cfg.explainer.rw_topk_edge = random.choice([300, 500, 700, 1000])
    cfg.explainer.reg_loss_inside = random.choice([1, 1.5, 2, 2.5, 3])
    cfg.explainer.reg_loss_outside = random.choice([0, 0.25, 0.5, 0.75, 1])

    cfg.wandb.name = cfg.wandb.name + f"_{args.hyper_run_id}"
    torch.save(cfg, f"hyperparam_cfg_rawexplainer.pt")
