#!/bin/bash

port=$(python script/find_free_port.py)

for i in {0..9}
do
    python script_hypersearch/hypersearch_rawexplainer_fb.py -c config/new_explain/fb15k237-rawexplainer-hyp-ola3.yaml --hyper_run_id="$i"
    python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/global_explanation_optim.py -c config/new_explain/fb15k237-rawexplainer-hyp-ola3.yaml --hyper_search --train_config config/new_nbfnet/fb15k237-finetune-ola.yaml
done
# python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/train_nbfnet.py -c config/nbfnet/fb15k237-005d-030m-ola.yaml --gpus [0,1,2,3] --use_wandb yes