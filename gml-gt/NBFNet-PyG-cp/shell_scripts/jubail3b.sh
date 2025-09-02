#!/bin/bash

port=$(python script/find_free_port.py)

python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/train_nbfnet.py -c config/nbfnet/wn18rr-dist-0025d-020M-jubail.yaml --gpus [0,1,2,3] --use_wandb yes