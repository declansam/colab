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

for max_prob in 0.1 0.15 0.2 0.25 0.3
do
    python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/train_nbfnet.py -c config/nbfnet/wn18rr-dist-005d-ola.yaml --gpus [0,1,2,3] --use_wandb no --max_prob "$max_prob"
done

for max_prob in 0.1 0.15 0.2 0.25 0.3
do
    python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/train_nbfnet.py -c config/nbfnet/wn18rr-dist-010d-ola.yaml --gpus [0,1,2,3] --use_wandb no --max_prob "$max_prob"
done

