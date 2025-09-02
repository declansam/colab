#!/bin/bash

port=$(python script/find_free_port.py)
python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/global_explanation_optim.py -c config/new_explain/fb15k237-rawexplainer-inf-olab.yaml --gpus [0,1,2,3] --use_wandb yes --train_config config/new_nbfnet/fb15k237-finetune-inf-ola.yaml

# port=$(python script/find_free_port.py)

# for th in 0.5 0.4 0.3 0.6 0.7 0.2
# do
#     python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/global_explanation_optim.py -c config/new_explain/fb15k237-rawexplainer-inf-ola.yaml --th "$th" --gpus [0,1,2,3] --use_wandb yes --train_config config/new_nbfnet/fb15k237-finetune-inf-ola.yaml
# done
# python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/global_explanation_optim.py -c config/new_explain/fb15k237-pgexplainer-ola.yaml --use_wandb yes --gpus [0,1,2,3] --train_config config/new_nbfnet/fb15k237-finetune-ola.yaml

# port=$(python script/find_free_port.py)
# config=config/new_explain/fb15k237-pgexplainer-hyp-ola.yaml
# for i in {0..9}
# do
#     python script_hypersearch/hypersearch_pgexplainer_inductive.py -c "$config" --hyper_run_id="$i"
#     python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/global_explanation_optim.py -c "$config" --hyper_search
# done
# python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/global_explanation_optim.py -c config/new_explain/fb15k237-pgexplainer-ola.yaml --use_wandb yes --gpus [0,1,2,3] --train_config config/new_nbfnet/fb15k237-finetune-ola.yaml
