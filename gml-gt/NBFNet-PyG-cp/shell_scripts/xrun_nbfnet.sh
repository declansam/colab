#!/bin/bash

#Define the resource requirements here using #SBATCH

#SBATCH -o %j.out

#For requesting 10 CPUs
#SBATCH -q bosco
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
python -m torch.distributed.launch --nproc_per_node=2 script/xrun.py -c config/transductive/fb15k237-0.15b-inf.yaml --gpus [0,1]
# python -m torch.distributed.launch --nproc_per_node=2 script/xrun.py -c config/transductive/fb15k237-0.05b-inf.yaml --gpus [0,1]
# python -m torch.distributed.launch --nproc_per_node=2 script/xrun.py -c config/transductive/fb15k237-0.1b-inf.yaml --gpus [0,1]
# python -m torch.distributed.launch --nproc_per_node=2 script/xrun.py -c config/transductive/fb15k237-0.2b-inf.yaml --gpus [0,1]
# python -m torch.distributed.launch --nproc_per_node=2 script/xrun.py -c config/transductive/fb15k237-0.3b-inf.yaml --gpus [0,1]