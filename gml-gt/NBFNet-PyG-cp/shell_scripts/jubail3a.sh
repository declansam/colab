#!/bin/bash

port=$(python script/find_free_port.py)

for check in /scratch/rk3570/Graph-Transformer/NBFNet-PyG/experiments/DistEval/NBFNet/FB15k-237/2025-03-06-15-29-43/model_epoch_4.pth
do
    python -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" script/train_nbfnet.py -c config/nbfnet/fb15k237-disteval-inf-jubail.yaml --gpus [0,1] --use_wandb no --check "$check"
done