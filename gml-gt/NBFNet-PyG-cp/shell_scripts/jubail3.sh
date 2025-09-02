#!/bin/bash

#Define the resource requirements here using #SBATCH

#SBATCH -o %j.out

#For requesting 10 CPUs
#SBATCH -q bosco
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

port=$(python script/find_free_port.py)
for k in 25 50 75 100 300 500 1000
do
    python -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" script/train_nbfnet.py -c config/new_nbfnet/fb15k237-finetune-inf-jub.yaml --use_wandb no --gpus [0,1] --k "$k"
done