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

config="config/explain/wn18rr-pagelink-hypersearch-jubail2.yaml"

for i in {0..19}
do
    python script_hypersearch/hypersearch_pagelink.py -c "$config" --hyper_run_id="$i"
    python script/instance_explanation.py -c "$config" --hyper_search
done