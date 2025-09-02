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
config="config/nbfnet/nell995-dist-jubail.yaml"

for d in 1 2 3 4 5 6 7 8 9 10
do
    python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/train_nbfnet.py -c "$config" --gpus [0,1,2,3] --use_wandb no --d "$d"
done

