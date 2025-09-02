#!/bin/bash

port=$(python script/find_free_port.py)

python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/train_nbfnet.py -c config/nbfnet/fb15k237-005d-015m.yaml --gpus [0,1,2,3] --use_wandb yes


