#!/bin/bash

# port=$(python script/find_free_port.py)

# python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/instance_explanation_optim.py -c config/new_explain/fb15k237-gnnexplainer-jubail.yaml --gpus [0,1,2,3] --use_wandb yes --train_config config/new_nbfnet/fb15k237-retrain-jubail.yaml