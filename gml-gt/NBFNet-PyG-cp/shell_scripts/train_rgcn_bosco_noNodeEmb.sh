#!/bin/bash

#Define the resource requirements here using #SBATCH

#SBATCH -o %j.out

#For requesting 10 CPUs
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 10
#SBATCH --mem=200GB

#Max wallTime for the job
#SBATCH -t 96:00:00

# echo Resource Allocated

#activate any environments if required
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
#activate any environments if required
conda activate Ultra

# echo Conda Env Activated

config="config/hetero/synthetic-no-node-emb.yaml"

port=$(python find_free_port.py)

# echo Port Found!
for i in {0..29}
do
    python script/hypersearch_rgcn_noNodeEmb.py -c "$config" --hyper_run_id="$i"
    python script/rgcn.py --hyper_search -c "$config"

done