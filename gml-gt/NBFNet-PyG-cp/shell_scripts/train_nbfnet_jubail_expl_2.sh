#!/bin/bash

#Define the resource requirements here using #SBATCH

#SBATCH -o %j.out

#For requesting 10 CPUs
#SBATCH -p nvidia
#SBATCH --gres=gpu:2
#SBATCH -c 10
#SBATCH --mem=200GB

#Max wallTime for the job
#SBATCH -t 96:00:00

#activate any environments if required
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
#activate any environments if required
conda activate Ultra

port=$(python script/find_free_port.py)
cfg1="config/nbfnet/wn18rr-gnnexplainer-finetune-ola.yaml"
cfg2="config/nbfnet/wn18rr-cf-gnnexplainer-finetune-ola.yaml"
cfg3="config/nbfnet/wn18rr-pgexplainer-finetune-jubail.yaml"
cfg4="config/nbfnet/wn18rr-pagelink-padded-finetune-jubail.yaml"

for k in 100 300 500 1000 5000
do
    python -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" script/train_nbfnet.py -c "$cfg1" --gpus [0,1] --use_wandb no --k "$k" --num_runs 5
done

for k in 100 300 500 1000 5000
do
    python -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" script/train_nbfnet.py -c "$cfg2" --gpus [0,1] --use_wandb no --k "$k" --num_runs 5
done

# for k in 100 300 500 1000 5000
# do
#     python -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" script/train_nbfnet.py -c "$cfg3" --gpus [0,1] --use_wandb no --k "$k" --num_runs 5
# done

# for k in 100 300 500 1000 5000
# do
#     python -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" script/train_nbfnet.py -c "$cfg4" --gpus [0,1] --use_wandb no --k "$k" --num_runs 5
# done