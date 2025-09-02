#!/bin/bash


port=$(python script/find_free_port.py)

for i in {0..9}
do
    python script_hypersearch/hypersearch_rawexplainer_fb.py -c config/new_explain/fb15k237-rawexplainer-hyp-ola5.yaml --hyper_run_id="$i"
    python -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" script/global_explanation_optim.py -c config/new_explain/fb15k237-rawexplainer-hyp-ola5.yaml --hyper_search --train_config config/new_nbfnet/fb15k237-finetune-ola5.yaml
done

# python -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" script/instance_explanation_optim.py -c config/new_explain/fb15k237-gnnexplainer-ola.yaml --gpus [0,1] --use_wandb yes --train_config config/new_nbfnet/fb15k237-finetune-ola.yaml

# python -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" script/instance_explanation_optim.py -c config/new_explain/wn18rr-gnnexplainer-ola.yaml --gpus [0,1] --use_wandb yes --train_config config/new_nbfnet/wn18rr-finetune-ola5.yaml

# port=$(python script/find_free_port.py)

# config=config/new_explain/wn18rr-rawexplainer-ola5.yaml

# for i in {0..19}
# do
#     python script_hypersearch/hypersearch_rawexplainer_with_path.py -c "$config" --hyper_run_id="$i"
#     python -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" script/global_explanation_optim.py -c "$config" --hyper_search --train_config config/new_nbfnet/wn18rr-finetune-ola5.yaml
# done