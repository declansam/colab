#!/bin/bash

port=$(python script/find_free_port.py)
config="config/nbfnet/wn18rr-rawexplainer-expl.yaml"

for k in 100 300 500 1000 5000
do
    python -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" script/train_nbfnet.py -c "$config" --gpus [1,2] --use_wandb no --k "$k"
done

