#!/bin/bash

#Define the resource requirements here using #SBATCH

#SBATCH -o %j.out

#For requesting 10 CPUs
#SBATCH -p nvidia
#SBATCH --gres=gpu:4
#SBATCH -c 10
#SBATCH --mem=200GB

#Max wallTime for the job
#SBATCH -t 96:00:00

#activate any environments if required
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
#activate any environments if required
conda activate Ultra

# find the free port first
port=$(python find_free_port.py)

# config_filtering
# python -m torch.distributed.launch --nproc_per_node=2 script/run_ego.py -c config/transductive/wn18rr-ego-0.1b.yaml --gpus [0,1] --use_wandb yes
python -m torch.distributed.launch --nproc_per_node=4 --master_port=$port script/samplex_ego.py -c config/transductive/wn18rr-ego-relpgexplainer-ego-nbf-Temp3-1.yaml --gpus [0,1,2,3] --use_wandb yes