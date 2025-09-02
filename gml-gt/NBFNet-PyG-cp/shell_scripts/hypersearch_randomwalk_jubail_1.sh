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

port=$(python find_free_port.py)
# config="config/transductive/wn18rr-kgsampling-hypersearch.yaml"
# config="config/hetero/synthetic-kgsampling-randomwalk-cf-hypersearch.yaml"
config="config/hetero/synthetic-kgsampling-gnnexplainer-cf-hypersearch.yaml"

for i in {0..29}
do
    # python script/hypersearch_randomwalk.py -c "$config" --hyper_run_id="$i"
    python script/hypersearch_gnnexplainer.py -c "$config" --hyper_run_id="$i"
    python script/instance_explanation.py -c "$config" --hyper_search
done