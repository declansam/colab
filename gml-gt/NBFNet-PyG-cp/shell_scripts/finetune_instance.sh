#!/bin/bash

port=$(python script/find_free_port.py)
# python -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" script/instance_explanation_optim.py -c config/explain/wn18rr-randomwalk-benchmark-jubail.yaml --train_config config/nbfnet/wn18rr-rawexplainer-finetune-jubail.yaml --gpus [0,1] --use_wandb no
# python -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" script/instance_explanation_optim.py -c config/explain/wn18rr-gnnexplainer-benchmark-jubail.yaml --train_config config/nbfnet/wn18rr-rawexplainer-finetune-jubail.yaml --gpus [0,1] --use_wandb no
# python -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" script/instance_explanation_optim.py -c config/explain/wn18rr-cf-gnnexplainer-benchmark-jubail.yaml --train_config config/nbfnet/wn18rr-rawexplainer-finetune-jubail.yaml --gpus [0,1] --use_wandb no
