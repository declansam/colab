#!/bin/bash

#Define the resource requirements here using #SBATCH

#SBATCH -o %j.out

#For requesting 10 CPUs
#SBATCH -C 80g
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:2
#SBATCH -c 10
#SBATCH --mem=200GB

#Max wallTime for the job
#SBATCH -t 96:00:00

#activate any environments if required
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
#activate any environments if required
conda activate Ultra

# config_filtering
python -m torch.distributed.launch --nproc_per_node=2 script/run_ego.py -c config/transductive/wn18rr-ego-0.2b.yaml --gpus [0,1] --use_wandb yes