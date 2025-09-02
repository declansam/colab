#!/bin/bash

#Define the resource requirements here using #SBATCH

#SBATCH -o %j.out

#For requesting 10 CPUs
#SBATCH -C 80g
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:4
#SBATCH -c 10
#SBATCH --mem=200GB

#Max wallTime for the job
#SBATCH -t 96:00:00

#activate any environments if required
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
#activate any environments if required
conda activate Ultra

port=$(python script/find_free_port.py)

python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/global_explanation_optim.py -c config/explain/fb15k237-rawexplainer-benchmark-jubail.yaml --gpus [0,1,2,3] --use_wandb no
python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/global_explanation_optim.py -c config/explain/fb15k237-pgexplainer-benchmark-jubail.yaml --gpus [0,1,2,3] --use_wandb no
python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/global_explanation_optim.py -c config/explain/fb15k237-xgexplainer-benchmark-jubail.yaml --gpus [0,1,2,3] --use_wandb no