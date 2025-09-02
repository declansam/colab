#!/bin/bash

#Define the resource requirements here using #SBATCH

#SBATCH -o %j.out

#For requesting 10 CPUs
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=200GB

#Max wallTime for the job
#SBATCH -t 96:00:00

#activate any environments if required
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
#activate any environments if required
conda activate Ultra

config1="config/explain/wn18rr-pagelink-benchmark-jubail3.yaml"
# config2="config/explain/wn18rr-pgexplainer-benchmark-jubail.yaml"

python script/instance_explanation_optim.py -c "$config1" --gpus [0] --use_wandb no