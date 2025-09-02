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
config="config/explain/wn18rr-rawexplainer-hyp-jubail4.yaml"
train_config="config/nbfnet/wn18rr-rawexplainer-finetune-jubail.yaml"

for i in {0..29}
do
    python script_hypersearch/hypersearch_rawexplainer_wn18rr_no_path.py -c "$config" --hyper_run_id="$i"
    python -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" script/global_explanation_optim.py -c "$config" --hyper_search --train_config "$train_config"
done