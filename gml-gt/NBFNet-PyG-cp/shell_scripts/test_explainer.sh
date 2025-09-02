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
config1="config/explain/wn18rr-gnnexplainer-benchmark-ola.yaml"
# config2="config/explain/wn18rr-pgexplainer-benchmark-jubail.yaml"

python -m torch.distributed.launch --nproc_per_node=4 script/instance_explanation_optim.py -c "$config1" --gpus [0,1,2,3] --use_wandb no

# python -m torch.distributed.launch --nproc_per_node=2 script/instance_explanation_optim.py -c "$config1" --gpus [0,1] --use_wandb no
# python -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" script/global_explanation_optim.py -c "$config2" --gpus [0,1] --use_wandb no