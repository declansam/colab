#!/bin/bash

port=$(python script/find_free_port.py)
config1="config/nbfnet/wn18rr-xgexplainer.yaml"

for k in 100 300 500 1000 5000
do
    python -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" script/train_nbfnet.py -c "$config1" --gpus [0,1] --use_wandb no --k "$k"
done

