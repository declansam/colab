#!/bin/bash

#Define the resource requirements here using #SBATCH

#SBATCH -o %j.out

#For requesting 10 CPUs
#SBATCH -q bosco
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 10
#SBATCH --mem=200GB

#Max wallTime for the job
#SBATCH -t 96:00:00

# echo Resource Allocated

#activate any environments if required
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
#activate any environments if required
conda activate Ultra

# echo Conda Env Activated

config="config/hetero/synthetic-randomized-edge-drop.yaml"

port=$(python find_free_port.py)

# echo Port Found!
for b in 0.1 0.15 0.2 0.25 0.3 0.35
do
    python script/rgcn.py --eval_on_edge_drop --randomized_edge_drop "$b" --use_wandb yes --gpus [0] -c "$config"
done