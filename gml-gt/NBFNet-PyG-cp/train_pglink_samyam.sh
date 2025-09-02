#!/bin/bash
#SBATCH --job-name=PGLink
#SBATCH --nodes=1
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200GB
#SBATCH --time=24:59:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slamichhane@nyu.edu
#SBATCH --output=./%x_%j.out
#SBATCH --error=./%x_%j.out


# python script/instance_explanation_optim.py -c config/new_explain/fb15k237-pagelink-jub-p0-new.yaml --gpus [0] --use_wandb no
code_dir=/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG
conda_env_path=/scratch/sl9030/conda-envs/ultra
python_path=$conda_env_path/bin/python
SCRIPT_NAME=script/instance_explanation_optim.py

# Logging
echo "Running script: $SCRIPT_NAME"
echo "Using Python interpreter: $python_path"
echo "Code directory: $code_dir"
echo "Conda environment path: $conda_env_path"
echo "Job started at: $(date)"

# Run script
$python_path $code_dir/$SCRIPT_NAME -c config/new_explain/fb15k237-pagelink-jub-p0-new.yaml --gpus [0] --use_wandb no

# Logging
echo "Job finished at: $(date)"
