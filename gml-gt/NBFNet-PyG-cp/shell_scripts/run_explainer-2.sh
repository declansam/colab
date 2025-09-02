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

port=$(python script/find_free_port.py)

python -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" script/global_explanation_optim.py -c config/explain/wn18rr-rawexplainer-val-benchmark-ola.yaml --gpus [0,1] --use_wandb yes --train_config config/nbfnet/wn18rr-rawexplainer-finetune-ola.yaml
# python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/global_explanation_optim.py -c config/explain/wn18rr-rawexplainer-ablation-f-dd.yaml --gpus [0,1,2,3] --use_wandb yes
# python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/global_explanation_optim.py -c config/explain/wn18rr-rawexplainer-ablation-cf-dd.yaml --gpus [0,1,2,3] --use_wandb yes
# python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/global_explanation_optim.py -c config/explain/wn18rr-rawexplainer-ablation-gnn_eval-dd.yaml --gpus [0,1,2,3] --use_wandb yes
# python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/global_explanation_optim.py -c config/explain/wn18rr-rawexplainer-ablation-gnn_expl-dd.yaml --gpus [0,1,2,3] --use_wandb yes
