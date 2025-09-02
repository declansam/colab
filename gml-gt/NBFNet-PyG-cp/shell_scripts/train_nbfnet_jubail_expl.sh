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
cfg1="config/nbfnet/wn18rr-rawexplainer-ablation-rw-dd.yaml"
cfg2="config/nbfnet/wn18rr-rawexplainer-ablation-f-dd.yaml"
cfg3="config/nbfnet/wn18rr-rawexplainer-ablation-cf-dd.yaml"
cfg4="config/nbfnet/wn18rr-rawexplainer-ablation-gnn_eval-dd.yaml"
cfg5="config/nbfnet/wn18rr-rawexplainer-ablation-gnn_expl-dd.yaml"

for k in 100 300 500 1000 5000
do
    python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/train_nbfnet.py -c "$cfg1" --gpus [0,1,2,3] --use_wandb no --k "$k"
    python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/train_nbfnet.py -c "$cfg2" --gpus [0,1,2,3] --use_wandb no --k "$k"
    python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/train_nbfnet.py -c "$cfg3" --gpus [0,1,2,3] --use_wandb no --k "$k"
    python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/train_nbfnet.py -c "$cfg4" --gpus [0,1,2,3] --use_wandb no --k "$k"
    python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/train_nbfnet.py -c "$cfg5" --gpus [0,1,2,3] --use_wandb no --k "$k"
done

