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

#activate any environments if required
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
#activate any environments if required
conda activate Ultra

port=$(python find_free_port.py)
# config="config/transductive/wn18rr-kgsampling-hypersearch.yaml"
# config="config/hetero/synthetic-kgsampling-relpgexplainer-noneg-cf-hypersearch.yaml"
config="config/hetero/synthetic-kgsampling-rwexplainer-noneg-cf-hypersearch.yaml"

for i in {0..29}
do
    # python script_hypersearch/hypersearch_relpgexplainer_synthetic.py -c "$config" --hyper_run_id="$i"
    python script_hypersearch/hypersearch_rwexplainer_synthetic.py -c "$config" --hyper_run_id="$i"
    # python script/hypersearch_rwexplainer.py -c "$config" --hyper_run_id="$i"
    # python -m torch.distributed.launch --nproc_per_node=3 --master_port="$port" script/kg_sampling.py -c "$config" --hyper_search
    python script/global_explanation.py -c "$config" --hyper_search
done