#!/bin/bash

for k in 25 50 75 100 300 500
do
    python script/count_components.py -c config/new_nbfnet/fb15k237-count-ola.yaml --gpus [0] --use_wandb no --k "$k"
done