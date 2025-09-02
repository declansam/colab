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

port=$(python script/find_free_port.py)

python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/train_nbfnet.py -c config/nbfnet/nell995.yaml --gpus [0,1,2,3] --use_wandb yes
python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/train_nbfnet.py -c config/nbfnet/nell995-005b.yaml --gpus [0,1,2,3] --use_wandb yes
python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/train_nbfnet.py -c config/nbfnet/nell995-010b.yaml --gpus [0,1,2,3] --use_wandb yes
python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/train_nbfnet.py -c config/nbfnet/nell995-015b.yaml --gpus [0,1,2,3] --use_wandb yes
python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/train_nbfnet.py -c config/nbfnet/nell995-020b.yaml --gpus [0,1,2,3] --use_wandb yes
