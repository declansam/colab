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

port=$(python find_free_port.py)
config="config/transductive/fb15k237-kgsampling-gnnexplainer-hypersearch-DD.yaml"

for i in {0..9}
do
    python script/hypersearch_gnnexplainer.py -c "$config" --hyper_run_id="$i"
    python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/instance_explanation.py -c "$config" --hyper_search
done